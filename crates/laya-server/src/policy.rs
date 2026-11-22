//! Parallax request admission and counters, based on deploy/laya/laya_server.py.

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use subtle::ConstantTimeEq;

use crate::{Predictor, Response, Server};

thread_local! {
    static DEADLINE: Cell<f64> = const { Cell::new(f64::INFINITY) };
    static EXPIRED: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn deadline() -> f64 {
    DEADLINE.get()
}

pub(crate) fn mark_expired() {
    EXPIRED.set(true);
}

pub(crate) fn expired(deadline: f64) -> bool {
    SystemTime::now().duration_since(UNIX_EPOCH).is_ok_and(|now| now.as_secs_f64() * 1000.0 > deadline)
}

#[derive(Default)]
struct Counts {
    answered: u64,
    busy: u64,
    expired: u64,
    unauthorized: u64,
    bad_request: u64,
    waiting: i64,
    max_waiting: i64,
}

/// Optional Parallax-compatible HTTP admission controls.
pub struct RequestPolicy {
    keys: Vec<Vec<u8>>,
    limit: Option<i64>,
    counts: Mutex<Counts>,
}

fn prediction(method: &str, target: &str) -> bool {
    method == "POST" && matches!(target, "/predict" | "/v1/decisions")
}

impl RequestPolicy {
    pub fn from_config(key: Option<&str>, limit: Option<i64>) -> Result<Self, String> {
        let env_key = std::env::var("SYSTEM1_LAYA_KEY").unwrap_or_default();
        let keys = key
            .filter(|s| !s.is_empty())
            .unwrap_or(&env_key)
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!("Bearer {s}").into_bytes())
            .collect::<Vec<_>>();
        let limit = match (limit, std::env::var("SYSTEM1_LAYA_MAX_QUEUE").ok()) {
            (Some(limit), _) => Some(limit),
            (None, Some(value)) => Some(value.trim().parse().map_err(|_| "SYSTEM1_LAYA_MAX_QUEUE needs an integer")?),
            (None, None) if !keys.is_empty() => Some(4),
            _ => None,
        };
        Ok(Self { keys, limit, counts: Mutex::default() })
    }

    pub fn requires_key(&self) -> bool {
        !self.keys.is_empty()
    }

    fn counts(&self) -> MutexGuard<'_, Counts> {
        self.counts.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub fn authorize(&self, method: &str, target: &str, authorization: Option<&[u8]>) -> Option<Response> {
        if self.requires_key() && (prediction(method, target) || (method, target) == ("GET", "/stats")) {
            let got = authorization.unwrap_or_default();
            let matched = self.keys.iter().fold(subtle::Choice::from(0), |matched, key| matched | got.ct_eq(key));
            if !bool::from(matched) {
                self.counts().unauthorized += 1;
                return Some(Response::error(401, "unauthorized"));
            }
        }
        None
    }

    pub fn body_error(&self, method: &str, target: &str, message: &str) -> Response {
        if prediction(method, target) {
            self.counts().bad_request += 1;
        }
        Response::error(400, message)
    }

    pub fn handle<P: Predictor>(&self, server: &Server<P>, method: &str, target: &str, body: &[u8], deadline: Option<f64>) -> Response {
        if (method, target) == ("GET", "/stats")
            && let Some(limit) = self.limit
        {
            let mut response = server.handle(method, target, body);
            let c = self.counts();
            response.body.pop();
            use std::fmt::Write;
            write!(response.body, r#", "requests": {{"answered": {}, "busy": {}, "expired": {}, "unauthorized": {}, "bad_request": {}, "waiting": {}, "max_waiting": {}, "max_queue": {limit}}}}}"#,
                c.answered, c.busy, c.expired, c.unauthorized, c.bad_request, c.waiting, c.max_waiting).expect("write to string");
            return response;
        }
        if !prediction(method, target) || self.limit.is_none() && deadline.is_none() {
            return server.handle(method, target, body);
        }
        // The Python server parses malformed JSON before admitting the request.
        if self.limit.is_some() && laya::pyjson::parse(body).is_err() {
            self.counts().bad_request += 1;
            return server.handle(method, target, body);
        }
        {
            let mut c = self.counts();
            if self.limit.is_some_and(|limit| c.waiting >= limit) {
                c.busy += 1;
                return Response::error(503, "busy");
            }
            c.waiting += 1;
            c.max_waiting = c.max_waiting.max(c.waiting);
        }
        let mut lease = Lease { policy: self, status: 500 };
        let deadline = deadline.unwrap_or(f64::INFINITY);
        let response = if expired(deadline) {
            Response::error(503, "expired")
        } else {
            struct Reset;
            impl Drop for Reset {
                fn drop(&mut self) {
                    DEADLINE.set(f64::INFINITY);
                    EXPIRED.set(false);
                }
            }
            let _reset = Reset;
            DEADLINE.set(deadline);
            let response = server.handle(method, target, body);
            if EXPIRED.get() { Response::error(503, "expired") } else { response }
        };
        lease.status = response.status;
        response
    }
}

struct Lease<'a> {
    policy: &'a RequestPolicy,
    status: u16,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut c = self.policy.counts();
        c.waiting -= 1;
        match self.status {
            200 => c.answered += 1,
            503 => c.expired += 1,
            _ => c.bad_request += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laya::PromptResult;
    use laya::question::{LayaError, Prompt};

    struct Unused;
    impl Predictor for Unused {
        fn predict_many(&self, _: &[Prompt]) -> Result<Vec<PromptResult>, LayaError> {
            panic!("test requests must stop before inference")
        }
    }

    #[test]
    fn parallax_response_bytes_and_counters() {
        let policy =
            RequestPolicy { keys: vec![b"Bearer first".to_vec(), b"Bearer second".to_vec()], limit: Some(2), counts: Mutex::default() };
        let server = Server::new(Unused, 0);
        let request = |body: &[u8], deadline| policy.handle(&server, "POST", "/predict", body, deadline);
        let denied = policy.authorize("GET", "/stats", Some(b"Bearer wrong")).unwrap();
        assert_eq!((denied.status, denied.body.as_str()), (401, r#"{"error": "unauthorized"}"#));
        assert!(policy.authorize("GET", "/health", None).is_none());
        assert!(policy.authorize("POST", "/predict", Some(b"Bearer second")).is_none());
        {
            let mut c = policy.counts();
            c.waiting = 2;
            c.max_waiting = 2;
        }
        assert_eq!(request(b"", None).status, 400);
        let busy = request(b"{}", None);
        assert_eq!((busy.status, busy.body.as_str()), (503, r#"{"error": "busy"}"#));
        policy.counts().waiting = 0;
        let expired = request(b"{}", Some(0.0));
        assert_eq!((expired.status, expired.body.as_str()), (503, r#"{"error": "expired"}"#));
        assert_eq!(request(br#"{"batch":{}}"#, None).status, 200);
        let stats = policy.handle(&server, "GET", "/stats", b"", None);
        assert!(stats.body.ends_with(
            r#", "requests": {"answered": 1, "busy": 1, "expired": 1, "unauthorized": 1, "bad_request": 1, "waiting": 0, "max_waiting": 2, "max_queue": 2}}"#
        ));
        let baseline = RequestPolicy { keys: vec![], limit: None, counts: Mutex::default() };
        assert_eq!(baseline.handle(&server, "GET", "/stats", b"", None), server.handle("GET", "/stats", b""));
    }
}
