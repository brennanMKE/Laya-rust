use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use laya::{Batch, Engine, EngineError, Model, Output, checkpoint};

use super::MergedModel;
use crate::Server;

struct Fake {
    rows: Arc<Mutex<Vec<usize>>>,
    first: Mutex<Option<mpsc::Receiver<()>>>,
}

impl Engine for Fake {
    fn forward(&self, batch: &Batch) -> Result<Output, EngineError> {
        self.rows.lock().unwrap().push(batch.len());
        if let Some(first) = self.first.lock().unwrap().take() {
            first.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        Ok(Output {
            logits: (0..batch.len()).map(|i| vec![0.0, (batch.seq(i).iter().sum::<u32>() % 7) as f32]).collect(),
            act_logits: vec![[1.0, 0.0]; batch.len()],
        })
    }
}

#[test]
fn merges_queued_requests_and_keeps_their_responses() {
    let Some(dir) = std::env::var_os("LAYA_CHECKPOINT").map(Into::into).or_else(|| checkpoint::find(None, "", "", "").ok()) else { return };
    let model = |rows, first| Model::new(&dir, Box::new(Fake { rows, first: Mutex::new(first) })).unwrap();
    let rows = Arc::new(Mutex::new(Vec::new()));
    let (release, first) = mpsc::channel();
    let predictor = MergedModel::new(model(rows.clone(), Some(first)), true, true).unwrap();
    let (submitted, queued) = mpsc::channel();
    *predictor.submitted.lock().unwrap() = Some(submitted);
    let server = Server::new(predictor, 100);
    let baseline = Server::new(model(Default::default(), None), 100);
    let body = |state: &str| format!(r#"{{"state":"{state}","questions":{{"q":{{"type":"noul","instructions":"safe?"}}}}}}"#);
    let invalid = r#"{"state":"invalid","questions":{"q":{"type":"choice","instructions":"pick","criteria":[]}}}"#.to_owned();
    let requests = [body("first"), body("shared"), body("unique"), body("shared"), invalid];
    let expected: Vec<_> = requests.iter().map(|body| baseline.handle("POST", "/predict", body.as_bytes())).collect();

    std::thread::scope(|scope| {
        let mut replies = Vec::new();
        for body in &requests {
            replies.push(scope.spawn(|| server.handle("POST", "/predict", body.as_bytes())));
            queued.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        release.send(()).unwrap();
        for (reply, expected) in replies.into_iter().zip(expected) {
            assert_eq!(reply.join().unwrap(), expected);
        }
    });
    assert_eq!(*rows.lock().unwrap(), [1, 2]);
    assert_eq!(server.cache.stats().misses, 5);
}

#[test]
fn drops_an_expired_job_before_the_next_engine_pass() {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{RequestPolicy, policy};

    let Some(dir) = std::env::var_os("LAYA_CHECKPOINT").map(Into::into).or_else(|| checkpoint::find(None, "", "", "").ok()) else { return };
    let rows = Arc::new(Mutex::new(Vec::new()));
    let (release, first) = mpsc::channel();
    let model = Model::new(&dir, Box::new(Fake { rows: rows.clone(), first: Mutex::new(Some(first)) })).unwrap();
    let predictor = MergedModel::new(model, false, true).unwrap();
    let (submitted, queued) = mpsc::channel();
    *predictor.submitted.lock().unwrap() = Some(submitted);
    let server = Server::new(predictor, 0);
    let policy = RequestPolicy::from_config(None, Some(2)).unwrap();
    let request = br#"{"state":"a frog","questions":{"q":{"type":"noul","instructions":"safe?"}}}"#;
    let ask = |deadline| policy.handle(&server, "POST", "/predict", request, deadline);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| ask(None));
        queued.recv_timeout(Duration::from_secs(5)).unwrap();
        let ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() * 1000.0 + 500.0;
        let second = scope.spawn(move || ask(Some(ms)));
        queued.recv_timeout(Duration::from_secs(5)).unwrap();
        let full = policy.handle(&server, "POST", "/predict", request, None);
        assert_eq!((full.status, full.body.as_str()), (503, r#"{"error": "busy"}"#));
        while !policy::expired(ms) {
            std::thread::sleep(Duration::from_millis(2));
        }
        release.send(()).unwrap();
        assert_eq!(first.join().unwrap().status, 200);
        let expired = second.join().unwrap();
        assert_eq!((expired.status, expired.body.as_str()), (503, r#"{"error": "expired"}"#));
    });
    assert_eq!(*rows.lock().unwrap(), [1]);
}
