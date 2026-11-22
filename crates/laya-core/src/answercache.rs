//! System 1 Arcade's `AnswerCache` and `answer_cached` (`agents/laya_batch.py`, laya-go's
//! `answercache`): a bounded LRU of answers keyed by [`crate::pydict::cache_key`], with the
//! same hit, miss and call counts, so `GET /stats` matches `laya_server.py`'s.
//!
//! Laya is deterministic, and games repeat sentences a lot, so only misses need the model,
//! still as one batch.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use crate::calibrate::round4;
use crate::pyjson::float_repr;

/// The number of answers kept by default.
pub const DEFAULT_SIZE: usize = 10_000;

/// The environment variable that sets the size; 0 turns the cache off.
pub const ENV: &str = "SYSTEM1_LAYA_CACHE";

/// `$SYSTEM1_LAYA_CACHE` as a number, or `default` when unset or blank (`cache_size_from_env`).
pub fn size_from_env(default: usize) -> Result<usize, String> {
    match std::env::var(ENV) {
        Ok(v) if !v.trim().is_empty() => {
            // A negative size disables the cache, as in Python.
            v.trim().parse::<i64>().map(|n| n.max(0) as usize).map_err(|e| format!("{ENV}={v:?}: {e}"))
        }
        _ => Ok(default),
    }
}

struct Inner<V> {
    /// key → (value, recency stamp)
    entries: HashMap<String, (V, u64)>,
    /// recency stamp → key; the first entry is the least recently used.
    order: BTreeMap<u64, String>,
    tick: u64,
    hits: u64,
    misses: u64,
    calls: u64,
    skipped: u64,
}

impl<V: Clone> Inner<V> {
    fn touch(&mut self, key: &str) -> Option<V> {
        let tick = self.tick + 1;
        let (v, stamp) = self.entries.get_mut(key)?;
        let old = std::mem::replace(stamp, tick);
        let v = v.clone();
        self.tick = tick;
        let k = self.order.remove(&old).unwrap();
        self.order.insert(tick, k);
        Some(v)
    }

    fn put(&mut self, key: String, v: V, size: usize) {
        self.tick += 1;
        if let Some((old_v, stamp)) = self.entries.get_mut(&key) {
            *old_v = v;
            let old = std::mem::replace(stamp, self.tick);
            self.order.remove(&old);
        } else {
            self.entries.insert(key.clone(), (v, self.tick));
        }
        self.order.insert(self.tick, key);
        while self.entries.len() > size {
            let (_, k) = self.order.pop_first().unwrap();
            self.entries.remove(&k);
        }
    }
}

/// A bounded LRU of answers with hit and miss counts. A size of 0 disables it.
pub struct AnswerCache<V> {
    size: usize,
    inner: Mutex<Inner<V>>,
}

/// `AnswerCache.stats()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub enabled: bool,
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    /// hits / (hits + misses) rounded to 4 decimals; 0 before any question.
    pub hit_rate: f64,
    pub model_calls: u64,
    pub skipped_calls: u64,
}

impl Stats {
    /// The stats as `laya_server.py` writes them: `json.dumps(cache.stats())`.
    pub fn to_python_json(&self) -> String {
        format!(
            r#"{{"enabled": {}, "size": {}, "capacity": {}, "hits": {}, "misses": {}, "hit_rate": {}, "model_calls": {}, "skipped_calls": {}}}"#,
            self.enabled,
            self.size,
            self.capacity,
            self.hits,
            self.misses,
            float_repr(self.hit_rate),
            self.model_calls,
            self.skipped_calls
        )
    }

    /// `AnswerCache.summary()`: `cache 12 hits, 3 misses (80.0%), 2 of 5 model calls skipped`.
    pub fn summary(&self) -> String {
        format!(
            "cache {} hits, {} misses ({:.1}%), {} of {} model calls skipped",
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
            self.skipped_calls,
            self.model_calls + self.skipped_calls
        )
    }
}

impl<V: Clone> AnswerCache<V> {
    pub fn new(size: usize) -> Self {
        AnswerCache {
            size,
            inner: Mutex::new(Inner { entries: HashMap::new(), order: BTreeMap::new(), tick: 0, hits: 0, misses: 0, calls: 0, skipped: 0 }),
        }
    }

    pub fn enabled(&self) -> bool {
        self.size > 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<V>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Answers a request whose questions have the cache keys `keys[p][q]`, sending only what
    /// the cache can't answer to `run`, in one call. This is `answer_cached`:
    ///
    /// - a question whose key is cached is a hit;
    /// - a question whose key an earlier question of this request asked is also a hit;
    /// - every other question is a miss, and `run` gets the misses as a mask (`miss[p][q]`)
    ///   and returns answers in the same shape; only masked entries are read.
    ///
    /// A request without misses skips `run` (a skipped call). When the cache is disabled, `run`
    /// gets every question and nothing is counted. The lock isn't held while `run` works.
    pub fn answer<E>(&self, keys: &[Vec<String>], run: impl FnOnce(&[Vec<bool>]) -> Result<Vec<Vec<V>>, E>) -> Result<Vec<Vec<V>>, E> {
        if !self.enabled() {
            let all: Vec<Vec<bool>> = keys.iter().map(|ks| vec![true; ks.len()]).collect();
            return run(&all);
        }
        let mut found: HashMap<&str, V> = HashMap::new();
        let mut pending: HashMap<&str, (usize, usize)> = HashMap::new();
        let mut order: Vec<&str> = Vec::new();
        let mut miss: Vec<Vec<bool>> = keys.iter().map(|ks| vec![false; ks.len()]).collect();
        {
            let mut c = self.lock();
            for (p, ks) in keys.iter().enumerate() {
                for (q, k) in ks.iter().enumerate() {
                    if found.contains_key(k.as_str()) || pending.contains_key(k.as_str()) {
                        c.hits += 1;
                    } else if let Some(v) = c.touch(k) {
                        c.hits += 1;
                        found.insert(k, v);
                    } else {
                        c.misses += 1;
                        pending.insert(k, (p, q));
                        order.push(k);
                        miss[p][q] = true;
                    }
                }
            }
            if order.is_empty() {
                c.skipped += 1;
            } else {
                c.calls += 1;
            }
        }
        if !order.is_empty() {
            let got = run(&miss)?;
            let mut c = self.lock();
            for k in order {
                let (p, q) = pending[k];
                let v = got[p][q].clone();
                c.put(k.to_string(), v.clone(), self.size);
                found.insert(k, v);
            }
        }
        Ok(keys.iter().map(|ks| ks.iter().map(|k| found[k.as_str()].clone()).collect()).collect())
    }

    /// The counts so far.
    pub fn stats(&self) -> Stats {
        let c = self.lock();
        let asked = c.hits + c.misses;
        Stats {
            enabled: self.size > 0,
            size: c.entries.len(),
            capacity: self.size,
            hits: c.hits,
            misses: c.misses,
            hit_rate: if asked > 0 { round4(c.hits as f64 / asked as f64) } else { 0.0 },
            model_calls: c.calls,
            skipped_calls: c.skipped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(v: &[&[&str]]) -> Vec<Vec<String>> {
        v.iter().map(|p| p.iter().map(|s| s.to_string()).collect()).collect()
    }

    #[test]
    fn only_misses_run_and_lru_evicts() {
        let c = AnswerCache::<String>::new(2);
        let run = |miss: &[Vec<bool>]| -> Result<Vec<Vec<String>>, ()> {
            Ok(miss.iter().map(|p| p.iter().map(|&m| if m { "new".to_string() } else { String::new() }).collect()).collect())
        };
        let out = c.answer(&keys(&[&["a", "b", "a"]]), run).unwrap();
        assert_eq!(out, vec![vec!["new"; 3]]);
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.model_calls, s.size), (1, 2, 1, 2));
        c.answer(&keys(&[&["a"], &["b"]]), |_| -> Result<_, ()> { panic!("all cached") }).unwrap();
        assert_eq!(c.stats().skipped_calls, 1);
        c.answer(&keys(&[&["c"]]), run).unwrap(); // evicts "a", the least recently used
        let mut asked = None;
        c.answer(&keys(&[&["a", "b"]]), |m: &[Vec<bool>]| -> Result<_, ()> {
            asked = Some(m.to_vec());
            Ok(vec![vec!["x".to_string(), String::new()]])
        })
        .unwrap();
        assert_eq!(asked.unwrap(), vec![vec![true, false]]);
        assert_eq!(
            c.stats().to_python_json(),
            r#"{"enabled": true, "size": 2, "capacity": 2, "hits": 4, "misses": 4, "hit_rate": 0.5, "model_calls": 3, "skipped_calls": 1}"#
        );
    }
}
