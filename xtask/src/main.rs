//! Repository tasks: `cargo xtask <task>`.
//!
//! - `sync-fixtures [LAYA_GO_DIR]`: copy laya-go's `testdata/golden` into `tests/golden` and
//!   record laya-go's commit in `LAYA_GO_COMMIT`. Never regenerate fixtures here; regenerate
//!   them in laya-go when the reference or the checkpoint changes, then sync.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("sync-fixtures") => sync_fixtures(args.get(1).map(String::as_str).unwrap_or("../laya-go")),
        Some("cross") if args.len() <= 2 => cross(args.get(1).map(String::as_str).unwrap_or("aarch64-unknown-linux-gnu.2.28")),
        _ => {
            eprintln!("usage: cargo xtask sync-fixtures [LAYA_GO_DIR] | cross [ZIGBUILD_TARGET]");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn cross(target: &str) -> Result<(), String> {
    if target.is_empty()
        || !target.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        || matches!(target, "." | "..")
    {
        return Err("invalid Zigbuild target".into());
    }
    let root = root();
    let mut cargo = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(&root)
        .args([
            "zigbuild",
            "--target",
            target,
            "--workspace",
            "--release",
            "--locked",
            "--tests",
            "--bins",
            "--message-format=json-render-diagnostics",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start cargo zigbuild: {e}"))?;

    let mut binaries = Vec::new();
    for line in BufReader::new(cargo.stdout.take().unwrap()).lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
            eprintln!("{line}");
            continue;
        };
        if let Some(rendered) = message["message"]["rendered"].as_str() {
            eprint!("{rendered}");
        }
        if message["reason"] == "compiler-artifact"
            && let Some(path) = message["executable"].as_str()
        {
            let name = message["target"]["name"].as_str().unwrap_or_default();
            if name != "xtask" && message["profile"]["test"] == true {
                let manifest = Path::new(message["manifest_path"].as_str().unwrap_or_default());
                let package = manifest.parent().and_then(Path::file_name).and_then(|s| s.to_str()).unwrap_or("workspace");
                let kind = message["target"]["kind"][0].as_str().unwrap_or("test");
                binaries.push((PathBuf::from(path), Some(format!("{package}--{kind}--{name}"))));
            } else if name == "laya-server" && message["target"]["kind"][0] == "bin" {
                binaries.push((PathBuf::from(path), None));
            }
        }
    }
    if !cargo.wait().map_err(|e| e.to_string())?.success() {
        return Err("cargo zigbuild failed".into());
    }
    if !binaries.iter().any(|(_, test)| test.is_none()) || !binaries.iter().any(|(_, test)| test.is_some()) {
        return Err("cargo zigbuild did not emit both the server and test executables".into());
    }

    let package = root.join("target/cross").join(target);
    let tests = package.join("parity-tests");
    fs::create_dir_all(&tests).map_err(|e| e.to_string())?;
    for (source, test) in binaries {
        let destination = test.map_or_else(|| package.join("laya-server"), |name| tests.join(name));
        fs::copy(&source, destination).map_err(|e| format!("copy {}: {e}", source.display()))?;
    }
    let mut fixtures = 0;
    copy_dir(&root.join("tests/golden"), &package.join("golden"), &mut fixtures)?;
    fs::copy(root.join("tools/laya_bench.py"), package.join("laya_bench.py")).map_err(|e| e.to_string())?;
    println!("Cross-build package: {}", package.display());
    Ok(())
}

fn sync_fixtures(go: &str) -> Result<(), String> {
    let root = root();
    let go = if Path::new(go).is_absolute() { PathBuf::from(go) } else { root.join(go) };
    let src = go.join("testdata/golden");
    if !src.join("meta.json").exists() {
        return Err(format!("{} has no golden fixtures", src.display()));
    }
    let out = Command::new("git").args(["-C", &go.to_string_lossy(), "rev-parse", "HEAD"]).output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("{} is not a git checkout", go.display()));
    }
    let commit = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let dirty = Command::new("git")
        .args(["-C", &go.to_string_lossy(), "status", "--porcelain", "--", "testdata/golden"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if dirty {
        eprintln!("warning: laya-go's testdata/golden has uncommitted changes");
    }

    let dst = root.join("tests/golden");
    if dst.exists() {
        fs::remove_dir_all(&dst).map_err(|e| format!("remove {}: {e}", dst.display()))?;
    }
    let mut n = 0;
    copy_dir(&src, &dst, &mut n)?;
    fs::write(root.join("LAYA_GO_COMMIT"), format!("{commit}\n")).map_err(|e| e.to_string())?;
    println!("copied {n} files from {} at {commit}", src.display());
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path, n: &mut usize) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("mkdir {}: {e}", dst.display()))?;
    let mut entries: Vec<_> = fs::read_dir(src).map_err(|e| e.to_string())?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let (from, to) = (e.path(), dst.join(e.file_name()));
        if from.is_dir() {
            copy_dir(&from, &to, n)?;
        } else {
            fs::copy(&from, &to).map_err(|err| format!("copy {}: {err}", from.display()))?;
            *n += 1;
        }
    }
    Ok(())
}
