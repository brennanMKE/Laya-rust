//! One inference worker collects work queued during the previous pass.

use std::sync::{Arc, mpsc};

#[cfg(test)]
mod tests;

use laya::question::{LayaError, Named, Prompt, State};
use laya::{Model, PromptResult};

use crate::{Predictor, policy};

type ResultRow = Result<Vec<PromptResult>, LayaError>;

struct Job {
    prompts: Vec<Prompt>,
    done: mpsc::Sender<Option<ResultRow>>,
    deadline: f64,
}

/// A predictor that merges requests waiting for the model into its next forward pass.
pub struct MergedModel {
    model: Arc<Model>,
    queue: mpsc::Sender<Job>,
    #[cfg(test)]
    submitted: std::sync::Mutex<Option<mpsc::Sender<()>>>,
}

impl MergedModel {
    pub fn new(model: Model, deduplicate: bool, coalesce: bool) -> std::io::Result<Self> {
        let model = Arc::new(model);
        let worker = model.clone();
        let (queue, pending) = mpsc::channel::<Job>();
        std::thread::Builder::new().name("laya-inference".into()).spawn(move || {
            while let Ok(first) = pending.recv() {
                let (requests, waiters): (Vec<_>, Vec<_>) = std::iter::once(first)
                    .chain(pending.try_iter().take(if coalesce { usize::MAX } else { 0 }))
                    .filter_map(|job| {
                        if policy::expired(job.deadline) {
                            let _ = job.done.send(None);
                            None
                        } else {
                            Some((job.prompts, job.done))
                        }
                    })
                    .unzip();
                for (done, result) in waiters.into_iter().zip(worker.predict_grouped(&requests, deduplicate)) {
                    let _ = done.send(Some(result));
                }
            }
        })?;
        Ok(Self {
            model,
            queue,
            #[cfg(test)]
            submitted: Default::default(),
        })
    }
}

impl Predictor for MergedModel {
    fn predict_many(&self, prompts: &[Prompt]) -> ResultRow {
        let disconnected = |_| LayaError::BadRequest("inference worker stopped".into());
        let (done, reply) = mpsc::channel();
        self.queue.send(Job { prompts: prompts.to_vec(), done, deadline: policy::deadline() }).map_err(disconnected)?;
        #[cfg(test)]
        if let Some(submitted) = &*self.submitted.lock().unwrap() {
            let _ = submitted.send(());
        }
        match reply.recv().map_err(|_| LayaError::BadRequest("inference worker stopped".into()))? {
            Some(result) => result,
            None => {
                policy::mark_expired();
                Err(LayaError::BadRequest("expired".into()))
            }
        }
    }

    fn sequence_length(&self, state: &State, question: &Named) -> Result<usize, LayaError> {
        self.model.sequence_length(state, question)
    }
}
