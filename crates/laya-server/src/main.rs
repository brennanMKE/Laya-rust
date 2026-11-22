//! `laya-server`: Laya as a decision endpoint, a drop-in for System 1 Arcade's
//! `agents/laya_server.py` and laya-go's `laya-server`. When it is listening it prints
//!
//! ```text
//! agent ready http://<host>:<port>/predict
//! ```
//!
//! Usage:
//!
//! ```text
//! laya-server [--port 8000] [--host 127.0.0.1 | --addr host:port]
//!             [--model <checkpoint dir>] [--revision <rev>]
//!             [--cache-size N] [--threads N] [--stats-every S] [--kernels gemm|portable]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::IntoResponse;
use laya::Model;
use laya::answercache;
use laya::checkpoint;
use laya_server::{MAX_BODY, MergedModel, RequestPolicy, Server};

const USAGE: &str = "Usage: laya-server [flags]

Serve Laya decisions over HTTP, compatible with System 1 Arcade's agents/laya_server.py.

Flags:
  --host HOST         address to listen on (default 127.0.0.1; 0.0.0.0 to serve other machines)
  --port N            port to listen on (default 8000; 0 = any free port)
  --addr HOST:PORT    listen address; overrides --host and --port
  --model DIR|REPO    checkpoint directory, or a Hugging Face repo in the local cache
                      (default: $SYSTEM1_LAYA_MODEL, else convaiinnovations/laya)
  --revision REV      revision of the repo (default: the pinned 1c5edc17a7ac…)
  --cache-size N      answers to cache, 0 = off (default: $SYSTEM1_LAYA_CACHE or 10000)
  --threads N         engine threads (0 = all cores)
  --kernels NAME      gemm (default) or portable (also $LAYA_KERNELS)
  --stats-every S     print cache stats every S seconds (0 = never)
  --key KEYS          comma-separated bearer keys for predictions and stats
  --max-queue N       limit requests admitted at once (default 4 when a key is set)
  --offline           never download checkpoints
  --device, --engine  accepted for compatibility with laya_server.py and laya-go; ignored
  --version           print the version and exit";

#[derive(Debug, Clone, PartialEq)]
struct Config {
    host: String,
    port: u16,
    addr: Option<String>,
    model: Option<String>,
    revision: Option<String>,
    cache_size: Option<usize>,
    threads: usize,
    kernels: Option<String>,
    stats_every: f64,
    weights: Option<String>,
    key: Option<String>,
    max_queue: Option<i64>,
    offline: bool,
    version: bool,
}

/// Parses flags the way Go's `flag` package does: `-name value`, `--name value`,
/// `-name=value`, and bare booleans.
fn parse_flags(args: &[String]) -> Result<Config, String> {
    let mut c = Config {
        host: "127.0.0.1".into(),
        port: 8000,
        addr: None,
        model: None,
        revision: None,
        cache_size: None,
        threads: 0,
        kernels: None,
        stats_every: 0.0,
        weights: None,
        key: None,
        max_queue: None,
        offline: false,
        version: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let Some(flag) = a.strip_prefix("--").or_else(|| a.strip_prefix('-')) else {
            return Err(format!("unexpected argument: {a}"));
        };
        let (name, inline) = match flag.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (flag, None),
        };
        let mut value = || inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| format!("flag needs an argument: -{name}"));
        let num = |v: String| v.parse::<f64>().map_err(|_| format!("invalid value {v:?} for flag -{name}"));
        match name {
            "h" | "help" => return Err(String::new()),
            "host" => c.host = value()?,
            "port" => c.port = num(value()?)? as u16,
            "addr" => c.addr = Some(value()?),
            "model" => c.model = Some(value()?),
            "revision" => c.revision = Some(value()?),
            "cache-size" => {
                let v = value()?;
                c.cache_size = Some(v.parse::<i64>().map_err(|_| format!("--cache-size {v:?}: not a number"))?.max(0) as usize);
            }
            "threads" => c.threads = num(value()?)? as usize,
            "kernels" => c.kernels = Some(value()?),
            "stats-every" => c.stats_every = num(value()?)?,
            "weights" => c.weights = Some(value()?),
            "key" => c.key = Some(value()?),
            "max-queue" => c.max_queue = Some(value()?.parse().map_err(|_| "--max-queue needs an integer")?),
            "device" | "engine" => {
                value()?;
            }
            "offline" => {
                c.offline = match inline.as_deref() {
                    None | Some("1" | "t" | "T" | "true" | "TRUE" | "True") => true,
                    Some("0" | "f" | "F" | "false" | "FALSE" | "False") => false,
                    Some(v) => return Err(format!("invalid value {v:?} for flag -offline")),
                };
            }
            "version" => c.version = true,
            _ => return Err(format!("flag provided but not defined: -{name}")),
        }
    }
    Ok(c)
}

/// The checkpoint directory for `--model` and `--revision`, the way `laya.load` reads its
/// argument: an existing path is a checkpoint directory, something that looks like a path but
/// doesn't exist is an error, and anything else is a Hugging Face repo id in the local cache.
fn checkpoint_dir(c: &Config) -> Result<PathBuf, String> {
    let mut m = c.model.clone().unwrap_or_else(|| std::env::var("SYSTEM1_LAYA_MODEL").unwrap_or_default().trim().to_string());
    if let Some(rest) = m.strip_prefix("~/") {
        m = Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest).to_string_lossy().into_owned();
    }
    let rev = c.revision.clone().unwrap_or_default();
    if m.is_empty() {
        return laya::hub::download(laya::hub::Options { revision: &rev, offline: c.offline, ..Default::default() })
            .map_err(|e| e.to_string());
    }
    let p = Path::new(&m);
    if p.exists() {
        checkpoint::check(p).map_err(|e| e.to_string())?;
        return Ok(p.into());
    }
    if m.starts_with('/') || m.starts_with("./") || m.starts_with("../") || p.is_absolute() {
        return Err(format!("local model path not found: {m:?}"));
    }
    let rev = if rev.is_empty() && m != checkpoint::REPO { "main".to_string() } else { rev };
    laya::hub::download(laya::hub::Options { repo: &m, revision: &rev, offline: c.offline, ..Default::default() })
        .map_err(|e| e.to_string())
}

fn kernel_set(c: &Config) -> Result<laya::KernelSet, String> {
    let name = c.kernels.clone().or_else(|| std::env::var("LAYA_KERNELS").ok()).unwrap_or_default();
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "gemm" | "auto" => Ok(laya::KernelSet::Gemm),
        "portable" => Ok(laya::KernelSet::Portable),
        other => Err(format!("unknown kernels {other:?} (want gemm or portable)")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let c = match parse_flags(&args) {
        Ok(c) => c,
        Err(e) if e.is_empty() => {
            println!("{USAGE}");
            return;
        }
        Err(e) => {
            eprintln!("laya-server: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if c.version {
        println!("laya-server {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if let Err(e) = run(c) {
        eprintln!("laya-server: {e}");
        std::process::exit(1);
    }
}

fn run(c: Config) -> Result<(), String> {
    if c.weights.is_some() {
        return Err("--weights (tuned heads from agents/finetune.py) is not supported yet".into());
    }
    let cache_size = match c.cache_size {
        Some(n) => n,
        None => answercache::size_from_env(answercache::DEFAULT_SIZE)?,
    };
    let policy = Arc::new(RequestPolicy::from_config(c.key.as_deref(), c.max_queue)?);
    let dir = checkpoint_dir(&c)?;
    let kernels = kernel_set(&c)?;
    let start = Instant::now();
    let model = Model::load(&dir, laya::Options { threads: c.threads, kernels }).map_err(|e| e.to_string())?;
    let info = model.info();
    eprintln!(
        "laya-server: loaded {} in {:.1}s (max_len {}, head_max_len {}, cache {cache_size})",
        dir.display(),
        start.elapsed().as_secs_f64(),
        info.max_len,
        info.head_max_len
    );
    eprintln!("laya-server: engine: {}", info.engine);

    let coalesce = std::env::var("LAYA_COALESCE").as_deref() != Ok("0");
    let model = MergedModel::new(model, cache_size > 0, coalesce).map_err(|e| e.to_string())?;
    let server = Arc::new(Server::new(model, cache_size));
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().map_err(|e| e.to_string())?;
    rt.block_on(serve(c, server, policy))
}

async fn serve(c: Config, server: Arc<Server<MergedModel>>, policy: Arc<RequestPolicy>) -> Result<(), String> {
    let addr = c
        .addr
        .clone()
        .unwrap_or_else(|| if c.host.contains(':') { format!("[{}]:{}", c.host, c.port) } else { format!("{}:{}", c.host, c.port) });
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| format!("listen {addr}: {e}"))?;
    let local = listener.local_addr().map_err(|e| e.to_string())?;
    // Print the host as given, like laya_server.py.
    let host = addr.rsplit_once(':').map(|(h, _)| h.to_string()).filter(|h| !h.is_empty()).unwrap_or_else(|| "0.0.0.0".into());
    let key_notice = if policy.requires_key() { " (key required)" } else { "" };
    println!("agent ready http://{host}:{}/predict{key_notice}", local.port());

    if c.stats_every > 0.0 {
        let s = server.clone();
        let every = Duration::from_secs_f64(c.stats_every);
        tokio::spawn(async move {
            let mut t = tokio::time::interval(every);
            t.tick().await;
            loop {
                t.tick().await;
                println!("{}", s.cache.stats().summary());
            }
        });
    }

    let app = axum::Router::new().fallback(move |req: Request<Body>| {
        let server = server.clone();
        let policy = policy.clone();
        async move { handle(server, policy, req).await }
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())
}

async fn handle(server: Arc<Server<MergedModel>>, policy: Arc<RequestPolicy>, req: Request<Body>) -> axum::response::Response {
    let method = req.method().as_str().to_string();
    let target = req.uri().path_and_query().map_or("/", |p| p.as_str()).to_string();
    if let Some(rejection) = policy.authorize(&method, &target, req.headers().get(header::AUTHORIZATION).map(|v| v.as_bytes())) {
        return http_response(rejection);
    }
    let deadline = req.headers().get("X-Deadline-Ms").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok());
    let body = match to_bytes(req.into_body(), MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return http_response(policy.body_error(&method, &target, &e.to_string())),
    };
    // The forward pass is CPU-bound and can take seconds: keep it off the async workers.
    let request = move || policy.handle(&server, &method, &target, &body, deadline);
    match tokio::task::spawn_blocking(request).await {
        Ok(r) => http_response(r),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!(r#"{{"error": "internal: {e}"}}"#)).into_response(),
    }
}

fn http_response(r: laya_server::Response) -> axum::response::Response {
    (StatusCode::from_u16(r.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), [(header::CONTENT_TYPE, r.content_type)], r.body)
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn flags() {
        let c = parse_flags(&args("-port 9000 --host=0.0.0.0 --cache-size 0 --device mps --engine metal --offline --threads 4")).unwrap();
        assert_eq!((c.port, c.host.as_str(), c.cache_size, c.threads), (9000, "0.0.0.0", Some(0), 4));
        assert!(c.offline);
        assert!(!parse_flags(&args("--offline=false")).unwrap().offline);
        assert!(parse_flags(&args("--nope")).is_err());
        assert!(parse_flags(&args("--port")).is_err());
        assert!(parse_flags(&args("stray")).is_err());
    }

    #[test]
    fn model_paths() {
        let mut c = parse_flags(&[]).unwrap();
        c.model = Some("./does-not-exist".into());
        assert_eq!(checkpoint_dir(&c).unwrap_err(), "local model path not found: \"./does-not-exist\"");
    }
}
