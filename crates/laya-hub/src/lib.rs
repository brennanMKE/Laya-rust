//! Resumable, verified checkpoint downloads using the Hugging Face cache layout.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use laya_core::checkpoint;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use url::Url;

pub use checkpoint::{REPO, REVISION, cache_dir, find};

const FILES: [&str; 5] =
    ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "model.safetensors"];

/// The checkpoint and cache to use. Empty repo/revision select the pinned checkpoint.
#[derive(Default)]
pub struct Options<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    pub cache: Option<&'a Path>,
    pub offline: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("hub: checkpoint not cached and downloads are disabled: {0}")]
    Offline(#[source] checkpoint::CheckpointError),
    #[error("hub: {0}")]
    Download(String),
    #[error("hub: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Checkpoint(#[from] checkpoint::CheckpointError),
}

/// Find a complete checkpoint or resume its download.
pub fn download(options: Options<'_>) -> Result<PathBuf, Error> {
    let endpoint = std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "https://huggingface.co".into());
    let token = std::env::var("HF_TOKEN").unwrap_or_default();
    let offline = options.offline
        || std::env::var("HF_HUB_OFFLINE").is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
    resolve(Options { offline, ..options }, &endpoint, &token, &FILES)
}

fn resolve(options: Options<'_>, endpoint: &str, token: &str, files: &[&str]) -> Result<PathBuf, Error> {
    let repo = if options.repo.is_empty() { REPO } else { options.repo };
    let revision = if options.revision.is_empty() { REVISION } else { options.revision };
    validate_path(repo)?;
    validate_path(revision)?;
    let cache = options.cache.map(Path::to_path_buf).unwrap_or_else(cache_dir);
    match find(Some(&cache), repo, revision, "") {
        Ok(path) => return Ok(path),
        Err(err) if options.offline => return Err(Error::Offline(err)),
        Err(_) => {}
    }
    let endpoint = Url::parse(endpoint).map_err(|e| Error::Download(format!("invalid endpoint: {e}")))?;
    let hub = Hub { endpoint, token, agent: ureq::AgentBuilder::new().redirects(0).build() };
    let base = cache.join(format!("models--{}", repo.replace('/', "--")));
    fs::create_dir_all(&base)?;
    let lock = OpenOptions::new().create(true).truncate(false).write(true).open(base.join(".laya-download.lock"))?;
    lock.lock()?;
    if let Ok(path) = find(Some(&cache), repo, revision, "") {
        return Ok(path);
    }
    let mut metadata = Vec::new();
    for file in files {
        metadata.push((file, hub.head(hub.url(repo, revision, file)?)?));
    }
    let commit = &metadata[0].1.commit;
    if (hex(revision, 40) && revision != commit) || metadata.iter().any(|(_, meta)| &meta.commit != commit) {
        return Err(Error::Download(format!("inconsistent checkout for {repo}@{revision}")));
    }
    let snapshot = base.join("snapshots").join(commit);
    for (file, meta) in &metadata {
        let blob = base.join("blobs").join(&meta.etag);
        if !blob.is_file() {
            hub.fetch(hub.url(repo, commit, file)?, meta, &blob)?;
        }
        link(&snapshot, file, &blob, &meta.etag)?;
    }
    checkpoint::check(&snapshot)?;
    if commit != revision {
        let reference = base.join("refs").join(revision);
        fs::create_dir_all(reference.parent().unwrap())?;
        let tmp = reference.with_extension("laya-ref-tmp");
        fs::write(&tmp, commit)?;
        fs::rename(&tmp, reference)?;
    }
    Ok(snapshot)
}

fn validate_path(value: &str) -> Result<(), Error> {
    if value.split('/').any(|part| {
        part.is_empty() || matches!(part, "." | "..") || !part.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
    }) {
        return Err(Error::Download(format!("invalid repo or revision {value:?}")));
    }
    Ok(())
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

struct Metadata {
    commit: String,
    etag: String,
    size: Option<u64>,
}

struct Hub<'a> {
    endpoint: Url,
    token: &'a str,
    agent: ureq::Agent,
}

impl Hub<'_> {
    fn url(&self, repo: &str, revision: &str, file: &str) -> Result<Url, Error> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut()
            .map_err(|()| Error::Download("endpoint has no path".into()))?
            .pop_if_empty()
            .extend(repo.split('/'))
            .push("resolve")
            .push(revision)
            .extend(file.split('/'));
        Ok(url)
    }

    fn request(&self, method: &str, url: &Url, range: Option<u64>) -> Result<ureq::Response, Error> {
        let mut request = self.agent.request(method, url.as_str()).set("Accept-Encoding", "identity");
        if !self.token.is_empty() && url.scheme() == "https" && url.host_str() == Some("huggingface.co") && url.port().is_none() {
            request = request.set("Authorization", &format!("Bearer {}", self.token));
        }
        if let Some(range) = range {
            request = request.set("Range", &format!("bytes={range}-"));
        }
        // Avoid putting redirect URLs, which can contain signed query strings, in error output.
        match request.call() {
            Ok(response) | Err(ureq::Error::Status(_, response)) => Ok(response),
            Err(_) => Err(Error::Download(format!("{method} request to {} failed", url.host_str().unwrap_or_default()))),
        }
    }

    fn redirect(&self, url: &Url, response: &ureq::Response) -> Result<Url, Error> {
        let next = response
            .header("Location")
            .and_then(|value| url.join(value).ok())
            .ok_or_else(|| Error::Download("redirect without a usable Location".into()))?;
        if !matches!(next.scheme(), "https" | "http") || url.scheme() == "https" && next.scheme() != "https" {
            return Err(Error::Download("invalid redirect protocol".into()));
        }
        Ok(next)
    }

    fn head(&self, mut url: Url) -> Result<Metadata, Error> {
        let mut commit = String::new();
        let mut etag = String::new();
        let mut size = None;
        for _ in 0..10 {
            let response = self.request("HEAD", &url, None)?;
            if commit.is_empty() {
                commit = response.header("X-Repo-Commit").unwrap_or_default().to_owned();
            }
            if etag.is_empty() {
                etag = clean_etag(response.header("X-Linked-Etag").unwrap_or_default()).to_owned();
            }
            size = size.or_else(|| response.header("X-Linked-Size")?.parse().ok());
            match response.status() {
                200 => {
                    if etag.is_empty() {
                        etag = clean_etag(response.header("ETag").unwrap_or_default()).to_owned();
                    }
                    size = size.or_else(|| response.header("Content-Length")?.parse().ok());
                }
                300..=399 => {
                    let next = self.redirect(&url, &response)?;
                    if next.origin() == url.origin() {
                        url = next;
                        continue;
                    }
                    // A CDN redirect carries the blob identity in X-Linked-Etag.
                }
                status => return Err(Error::Download(format!("HEAD returned HTTP {status}"))),
            }
            if !hex(&commit, 40) || !(hex(&etag, 40) || hex(&etag, 64)) {
                return Err(Error::Download("missing or invalid commit or checksum from the Hub".into()));
            }
            return Ok(Metadata { commit, etag, size });
        }
        Err(Error::Download("too many HEAD redirects".into()))
    }

    fn get(&self, mut url: Url, range: Option<u64>) -> Result<ureq::Response, Error> {
        for _ in 0..10 {
            let response = self.request("GET", &url, range)?;
            if !(300..400).contains(&response.status()) {
                return Ok(response);
            }
            url = self.redirect(&url, &response)?;
        }
        Err(Error::Download("too many GET redirects".into()))
    }

    fn fetch(&self, url: Url, meta: &Metadata, blob: &Path) -> Result<(), Error> {
        fs::create_dir_all(blob.parent().unwrap())?;
        let partial = blob.with_extension("incomplete");
        let mut file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&partial)?;
        let mut have = file.seek(SeekFrom::End(0))?;
        if meta.size.is_some_and(|size| have > size) {
            file.set_len(0)?;
            have = 0;
        }
        if meta.size != Some(have) {
            let response = self.get(url, (have != 0).then_some(have))?;
            match response.status() {
                200 => {
                    file.set_len(0)?;
                    have = 0;
                }
                206 if response.header("Content-Range").and_then(range_start) == Some(have) => {}
                416 => {}
                status => return Err(Error::Download(format!("unexpected HTTP {status} when downloading from byte {have}"))),
            }
            file.seek(SeekFrom::Start(have))?;
            if response.status() != 416 {
                have += io::copy(&mut response.into_reader(), &mut file)?;
            }
        }
        if meta.size.is_some_and(|size| have != size) {
            return Err(Error::Download(format!("download has {have} bytes, expected {}", meta.size.unwrap())));
        }
        let checksum = checksum(&mut file, &meta.etag, have)?;
        if checksum != meta.etag {
            drop(file);
            fs::remove_file(partial)?;
            return Err(Error::Download(format!("checksum mismatch: got {checksum}, expected {}", meta.etag)));
        }
        file.sync_all()?;
        drop(file);
        fs::rename(partial, blob)?;
        Ok(())
    }
}

fn clean_etag(value: &str) -> &str {
    let value = value.trim();
    value.strip_prefix("W/").unwrap_or(value).trim_matches('"')
}

fn range_start(value: &str) -> Option<u64> {
    value.strip_prefix("bytes ")?.split_once('-')?.0.parse().ok()
}

fn checksum(file: &mut File, etag: &str, size: u64) -> io::Result<String> {
    file.rewind()?;
    if etag.len() == 64 {
        let mut digest = Sha256::new();
        io::copy(file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    } else {
        let mut digest = Sha1::new();
        write!(digest, "blob {size}\0")?;
        io::copy(file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    }
}

fn link(snapshot: &Path, relative: &str, blob: &Path, etag: &str) -> Result<(), Error> {
    let destination = snapshot.join(relative);
    if destination.is_file() {
        return Ok(());
    }
    fs::create_dir_all(destination.parent().unwrap())?;
    let mut link_target = PathBuf::new();
    for _ in 0..=relative.split('/').count() {
        link_target.push("..");
    }
    link_target.push("blobs");
    link_target.push(etag);
    let temporary = destination.with_extension("laya-link-tmp");
    if temporary.symlink_metadata().is_ok() {
        fs::remove_file(&temporary)?;
    }
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&link_target, &temporary).is_ok();
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(&link_target, &temporary).is_ok();
    #[cfg(not(any(unix, windows)))]
    let linked = false;
    if !linked {
        fs::copy(blob, &temporary)?;
    }
    fs::rename(temporary, destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::mpsc;

    const DATA: &[u8] = b"0123456789";

    #[test]
    fn resume_and_verify_local_download() {
        let cache = tempfile::tempdir().unwrap();
        let etag = format!("{:x}", Sha256::digest(DATA));
        let blobs = cache.path().join("models--convaiinnovations--laya/blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join(format!("{etag}.incomplete")), &DATA[..4]).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (done, recv) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            for i in 0..6 {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
                let mut reader = BufReader::new(&socket);
                let mut line = String::new();
                let mut request = String::new();
                loop {
                    line.clear();
                    assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                    request.push_str(&line);
                    if line == "\r\n" {
                        break;
                    }
                }
                let lower = request.to_ascii_lowercase();
                assert!(!lower.contains("authorization:"));
                if i < 5 {
                    assert!(request.starts_with("HEAD "));
                    write!(socket, "HTTP/1.1 200 OK\r\nX-Repo-Commit: {REVISION}\r\nETag: \"{etag}\"\r\nContent-Length: 10\r\nConnection: close\r\n\r\n").unwrap();
                } else {
                    assert!(request.starts_with("GET "));
                    assert!(lower.contains("range: bytes=4-"));
                    socket.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-9/10\r\nContent-Length: 6\r\nConnection: close\r\n\r\n456789").unwrap();
                }
            }
            done.send(()).unwrap();
        });
        let endpoint = format!("http://{address}");
        let options = || Options { revision: "main", cache: Some(cache.path()), ..Default::default() };
        let path = resolve(options(), &endpoint, "never-forward-this-token", &FILES).unwrap();
        recv.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert_eq!(path, cache.path().join(format!("models--convaiinnovations--laya/snapshots/{REVISION}")));
        for file in FILES {
            assert_eq!(fs::read(path.join(file)).unwrap(), DATA);
        }
        assert_eq!(resolve(Options { offline: true, ..options() }, "unused", "", &FILES).unwrap(), path);
    }

    #[test]
    fn checksum_failure_discards_the_partial() {
        let cache = tempfile::tempdir().unwrap();
        let blob = cache.path().join("blob");
        let partial = blob.with_extension("incomplete");
        fs::write(&partial, b"corrupted").unwrap();
        let hub = Hub { endpoint: Url::parse("http://127.0.0.1:1").unwrap(), token: "", agent: ureq::Agent::new() };
        let meta = Metadata { commit: REVISION.into(), etag: format!("{:x}", Sha256::digest(DATA)), size: Some(9) };
        let error = hub.fetch(hub.endpoint.clone(), &meta, &blob).unwrap_err();
        assert!(error.to_string().contains("checksum mismatch"));
        assert!(!partial.exists());
        assert!(!blob.exists());
    }
}
