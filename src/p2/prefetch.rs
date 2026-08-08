//! Background CPU batch generation to overlap with GPU forward/backward.

use crate::domain::Split;
use crate::p2::data::TransitionSample;
use crate::p2::train::collect_batch_uncached;
use anyhow::Result;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Default in-flight work + ready slots (current grad-accum step + next step).
pub const DEFAULT_PREFETCH_QUEUE_DEPTH: usize = 8;

pub struct PrefetchRequest {
    pub curriculum: String,
    pub seed: u64,
    pub episode_start: u64,
    pub physical_batch: usize,
    pub split: Split,
}

/// A queued request tagged with its submission order, so results can be handed
/// back deterministically regardless of which worker finishes first.
struct SeqRequest {
    seq: u64,
    req: PrefetchRequest,
}

enum PrefetchMsg {
    /// Submission sequence number of the request, then its batch.
    Ready(u64, Result<Vec<TransitionSample>>),
}

struct WorkQueue {
    pending: Mutex<VecDeque<SeqRequest>>,
    notify: Condvar,
}

/// Pipelined batch prefetch with a small worker pool (parallel curriculum generation).
///
/// Batches are handed back in **submission order**, not completion order. The
/// workers finish in a nondeterministic order, so consuming results as they
/// arrive made the episode sequence — and therefore training itself —
/// irreproducible, which broke pause/resume equivalence. `ready` is a reorder
/// buffer keyed by submission sequence and `next_out` is the next index owed to
/// the caller.
pub struct BatchPrefetcher {
    cancelled: Arc<AtomicBool>,
    accepting: bool,
    work: Arc<WorkQueue>,
    result_rx: Receiver<PrefetchMsg>,
    ready: BTreeMap<u64, Result<Vec<TransitionSample>>>,
    submitted: u64,
    next_out: u64,
    workers: Vec<JoinHandle<()>>,
}

fn prefetch_worker_count() -> usize {
    std::env::var("TOFY_P2_PREFETCH_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| rayon::current_num_threads().clamp(2, 32))
}

fn prefetch_queue_depth() -> usize {
    std::env::var("TOFY_P2_PREFETCH_QUEUE_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PREFETCH_QUEUE_DEPTH)
        .max(2)
}

impl BatchPrefetcher {
    pub fn new() -> Self {
        Self::with_queue_depth(prefetch_queue_depth())
    }

    pub fn with_queue_depth(queue_depth: usize) -> Self {
        let depth = queue_depth.max(2);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancelled);
        let work = Arc::new(WorkQueue {
            pending: Mutex::new(VecDeque::new()),
            notify: Condvar::new(),
        });
        let (result_tx, result_rx) = sync_channel::<PrefetchMsg>(depth);
        let worker_count = prefetch_worker_count();
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let work = Arc::clone(&work);
            let cancelled = Arc::clone(&worker_cancel);
            let result_tx = result_tx.clone();
            workers.push(thread::spawn(move || {
                while !cancelled.load(Ordering::Relaxed) {
                    let req = {
                        let mut guard = work.pending.lock().expect("prefetch work queue");
                        while guard.is_empty() && !cancelled.load(Ordering::Relaxed) {
                            guard = work.notify.wait(guard).expect("prefetch work notify");
                        }
                        if cancelled.load(Ordering::Relaxed) && guard.is_empty() {
                            None
                        } else {
                            guard.pop_front()
                        }
                    };
                    if req.is_none() {
                        continue;
                    }
                    let SeqRequest { seq, req } = req.unwrap();
                    let batch = collect_batch_uncached(
                        &req.curriculum,
                        req.seed,
                        req.episode_start,
                        req.physical_batch,
                        req.split,
                        Some(&cancelled),
                    );
                    if cancelled.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = result_tx.send(PrefetchMsg::Ready(seq, batch));
                }
            }));
        }
        Self {
            cancelled,
            accepting: true,
            work,
            result_rx,
            ready: BTreeMap::new(),
            submitted: 0,
            next_out: 0,
            workers,
        }
    }

    /// Stop accepting work and detach workers without waiting for in-flight generation.
    pub fn shutdown(&mut self) {
        self.accepting = false;
        self.cancelled.store(true, Ordering::SeqCst);
        self.work.notify.notify_all();
        self.drain_results();
        self.ready.clear();
        // A replacement prefetcher re-submits from zero, so the ordering cursor
        // restarts with the queue it belongs to.
        self.submitted = 0;
        self.next_out = 0;
        let _ = self.workers.drain(..);
    }

    pub fn submit(&mut self, req: PrefetchRequest) -> Result<()> {
        if !self.accepting {
            return Err(anyhow::anyhow!("prefetch submit after shutdown"));
        }
        let seq = self.submitted;
        self.submitted += 1;
        {
            let mut guard = self.work.pending.lock().expect("prefetch work queue");
            guard.push_back(SeqRequest { seq, req });
        }
        self.work.notify.notify_one();
        Ok(())
    }

    pub fn submit_many(&mut self, reqs: &[PrefetchRequest]) -> Result<()> {
        if reqs.is_empty() {
            return Ok(());
        }
        if !self.accepting {
            return Err(anyhow::anyhow!("prefetch submit after shutdown"));
        }
        let start = self.submitted;
        self.submitted += reqs.len() as u64;
        {
            let mut guard = self.work.pending.lock().expect("prefetch work queue");
            for (offset, req) in reqs.iter().enumerate() {
                guard.push_back(SeqRequest {
                    seq: start + offset as u64,
                    req: req.clone(),
                });
            }
        }
        self.work.notify.notify_all();
        Ok(())
    }

    pub fn poll(&mut self) {
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                // Queue failures in order rather than dropping them. A swallowed error
                // silently shrank the ready queue with no log, so the pipeline lost a
                // slot per failure and `recv` could block on a batch that never comes.
                PrefetchMsg::Ready(seq, batch) => {
                    self.ready.insert(seq, batch);
                }
            }
        }
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    fn drain_results(&mut self) {
        while self.result_rx.try_recv().is_ok() {}
    }

    /// Next batch **in submission order**, blocking until that specific batch
    /// arrives even if later ones finished first.
    pub fn recv(&mut self) -> Result<Vec<TransitionSample>> {
        self.poll();
        loop {
            if let Some(batch) = self.ready.remove(&self.next_out) {
                self.next_out += 1;
                return batch;
            }
            match self
                .result_rx
                .recv()
                .map_err(|e| anyhow::anyhow!("prefetch recv: {e}"))?
            {
                PrefetchMsg::Ready(seq, batch) => {
                    self.ready.insert(seq, batch);
                }
            }
        }
    }

    /// Receive `n` prepared sample batches (blocks until each is ready).
    pub fn recv_n(&mut self, n: usize) -> Result<Vec<Vec<TransitionSample>>> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            out.push(self.recv()?);
        }
        Ok(out)
    }
}

impl Default for BatchPrefetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BatchPrefetcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Clone for PrefetchRequest {
    fn clone(&self) -> Self {
        Self {
            curriculum: self.curriculum.clone(),
            seed: self.seed,
            episode_start: self.episode_start,
            physical_batch: self.physical_batch,
            split: self.split,
        }
    }
}
