//! Background CPU batch generation to overlap with GPU forward/backward.

use crate::domain::Split;
use crate::p2::data::{
    compose_mixed_stream_batch, MixedStreamBatch, MixedStreamConfig, TransitionSample, V5DataSplit,
};
use crate::p2::train::{collect_batch_uncached, training_content_batch_digest};
use anyhow::{ensure, Result};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Default in-flight work + ready slots (current grad-accum step + next step).
pub const DEFAULT_PREFETCH_QUEUE_DEPTH: usize = 8;
pub const FOUNDATION_V2_PREFETCH_QUEUE_DEPTH: usize = 4;

pub struct PrefetchRequest {
    pub curriculum: String,
    pub seed: u64,
    pub episode_start: u64,
    pub physical_batch: usize,
    pub split: Split,
}

/// Inputs that define which deterministic batch stream a prefetcher serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefetchScope {
    pub curriculum: String,
    pub seed: u64,
    pub physical_batch: usize,
    pub split: Split,
}

impl From<&PrefetchRequest> for PrefetchScope {
    fn from(request: &PrefetchRequest) -> Self {
        Self {
            curriculum: request.curriculum.clone(),
            seed: request.seed,
            physical_batch: request.physical_batch,
            split: request.split,
        }
    }
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
    scope: Option<PrefetchScope>,
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
            scope: None,
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
        self.scope = None;
        let _ = self.workers.drain(..);
    }

    pub fn scope(&self) -> Option<&PrefetchScope> {
        self.scope.as_ref()
    }

    fn accept_scope(&mut self, request: &PrefetchRequest) -> Result<()> {
        let requested = PrefetchScope::from(request);
        match &self.scope {
            Some(active) if active != &requested => Err(anyhow::anyhow!(
                "prefetch scope changed without restart: active={active:?} requested={requested:?}"
            )),
            Some(_) => Ok(()),
            None => {
                self.scope = Some(requested);
                Ok(())
            }
        }
    }

    pub fn submit(&mut self, req: PrefetchRequest) -> Result<()> {
        if !self.accepting {
            return Err(anyhow::anyhow!("prefetch submit after shutdown"));
        }
        self.accept_scope(&req)?;
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
        for request in reqs {
            self.accept_scope(request)?;
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

struct MixedStreamWorkQueue {
    pending: Mutex<VecDeque<u64>>,
    notify: Condvar,
}

/// A deterministic mixed-stream batch plus the digest of its exact ordered
/// training rows and content masks. Workers prepare both together so the
/// training thread never re-hashes full frames.
pub struct PreparedMixedStreamBatch {
    batch: MixedStreamBatch,
    training_content_digest: [u8; 32],
}

impl PreparedMixedStreamBatch {
    pub fn into_parts(self) -> (MixedStreamBatch, [u8; 32]) {
        (self.batch, self.training_content_digest)
    }
}

/// Ordered, bounded foundation-v2 batch pipeline.
///
/// Every worker composes from the immutable `(config, progress, batch_index,
/// split)` key. Completion order is deliberately hidden: `recv_next` returns
/// only the next batch index owed to the training loop.
pub struct MixedStreamBatchPrefetcher {
    cancelled: Arc<AtomicBool>,
    accepting: bool,
    work: Arc<MixedStreamWorkQueue>,
    result_rx: Receiver<(u64, Result<PreparedMixedStreamBatch>)>,
    ready: BTreeMap<u64, Result<PreparedMixedStreamBatch>>,
    workers: Vec<JoinHandle<()>>,
    total_steps: u64,
    queue_depth: usize,
    outstanding: usize,
    next_to_submit: u64,
    next_to_receive: u64,
}

impl MixedStreamBatchPrefetcher {
    pub fn new(
        config: MixedStreamConfig,
        total_steps: usize,
        first_batch_index: u64,
        worker_count: usize,
    ) -> Result<Self> {
        Self::with_queue_depth(
            config,
            total_steps,
            first_batch_index,
            worker_count,
            FOUNDATION_V2_PREFETCH_QUEUE_DEPTH,
        )
    }

    pub fn with_queue_depth(
        config: MixedStreamConfig,
        total_steps: usize,
        first_batch_index: u64,
        worker_count: usize,
        queue_depth: usize,
    ) -> Result<Self> {
        ensure!(worker_count > 0, "foundation-v2 data workers must be > 0");
        ensure!(queue_depth > 0, "foundation-v2 prefetch depth must be > 0");
        config.validate()?;
        ensure!(
            first_batch_index <= total_steps as u64,
            "foundation-v2 prefetch start exceeds total steps"
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let work = Arc::new(MixedStreamWorkQueue {
            pending: Mutex::new(VecDeque::with_capacity(queue_depth)),
            notify: Condvar::new(),
        });
        // The number of submitted-but-unconsumed requests is bounded by
        // queue_depth, so this unbounded result channel is bounded by the same
        // invariant without risking shutdown deadlock on a full sender.
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let config = Arc::new(config);
        let total_steps_u64 = total_steps as u64;
        let mut workers = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let worker_cancelled = Arc::clone(&cancelled);
            let worker_work = Arc::clone(&work);
            let worker_result_tx = result_tx.clone();
            let worker_config = Arc::clone(&config);
            let spawned = thread::Builder::new()
                .name(format!("foundation-v2-data-{worker_index}"))
                .spawn(move || loop {
                    let batch_index = {
                        let mut guard = worker_work
                            .pending
                            .lock()
                            .expect("mixed prefetch work queue");
                        while guard.is_empty() && !worker_cancelled.load(Ordering::Relaxed) {
                            guard = worker_work
                                .notify
                                .wait(guard)
                                .expect("mixed prefetch notify");
                        }
                        if worker_cancelled.load(Ordering::Relaxed) {
                            None
                        } else {
                            guard.pop_front()
                        }
                    };
                    let Some(batch_index) = batch_index else {
                        break;
                    };
                    let progress = batch_index as f32 / total_steps_u64.max(1) as f32;
                    let composed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<PreparedMixedStreamBatch> {
                            let batch = compose_mixed_stream_batch(
                                &worker_config,
                                progress,
                                batch_index,
                                V5DataSplit::Train,
                            )?;
                            let training_content_digest = training_content_batch_digest(
                                batch.transitions(),
                                batch.content_masks(),
                            )?;
                            Ok(PreparedMixedStreamBatch {
                                batch,
                                training_content_digest,
                            })
                        },
                    ))
                    .unwrap_or_else(|_| {
                        Err(anyhow::anyhow!(
                            "foundation-v2 data worker panicked at batch {batch_index}"
                        ))
                    });
                    if worker_cancelled.load(Ordering::Relaxed) {
                        break;
                    }
                    if worker_result_tx.send((batch_index, composed)).is_err() {
                        break;
                    }
                });
            match spawned {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    cancelled.store(true, Ordering::SeqCst);
                    work.notify.notify_all();
                    for worker in workers.drain(..) {
                        let _ = worker.join();
                    }
                    return Err(error.into());
                }
            }
        }
        drop(result_tx);
        let mut prefetcher = Self {
            cancelled,
            accepting: true,
            work,
            result_rx,
            ready: BTreeMap::new(),
            workers,
            total_steps: total_steps_u64,
            queue_depth,
            outstanding: 0,
            next_to_submit: first_batch_index,
            next_to_receive: first_batch_index,
        };
        prefetcher.refill();
        Ok(prefetcher)
    }

    fn refill(&mut self) {
        let mut guard = self.work.pending.lock().expect("mixed prefetch work queue");
        while self.accepting
            && self.outstanding < self.queue_depth
            && self.next_to_submit < self.total_steps
        {
            guard.push_back(self.next_to_submit);
            self.next_to_submit += 1;
            self.outstanding += 1;
        }
        drop(guard);
        self.work.notify.notify_all();
    }

    pub fn recv_next(&mut self) -> Result<(u64, PreparedMixedStreamBatch)> {
        loop {
            if let Some(batch) = self.ready.remove(&self.next_to_receive) {
                let batch_index = self.next_to_receive;
                self.next_to_receive += 1;
                self.outstanding -= 1;
                self.refill();
                return batch.map(|batch| (batch_index, batch));
            }
            let (batch_index, batch) = self
                .result_rx
                .recv()
                .map_err(|error| anyhow::anyhow!("foundation-v2 prefetch recv: {error}"))?;
            ensure!(
                self.ready.insert(batch_index, batch).is_none(),
                "foundation-v2 prefetch produced duplicate batch {batch_index}"
            );
        }
    }

    pub fn shutdown(&mut self) {
        if !self.accepting && self.workers.is_empty() {
            return;
        }
        self.accepting = false;
        self.cancelled.store(true, Ordering::SeqCst);
        self.work
            .pending
            .lock()
            .expect("mixed prefetch work queue")
            .clear();
        self.work.notify.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        while self.result_rx.try_recv().is_ok() {}
        self.ready.clear();
        self.outstanding = 0;
    }
}

impl Drop for MixedStreamBatchPrefetcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod mixed_stream_tests {
    use super::*;
    use crate::p2::data::foundation_v2_stream_schedule;
    use crate::p2::train::training_content_hash_append;

    #[test]
    fn mixed_stream_prefetch_is_ordered_and_uses_index_progress_on_resume() -> Result<()> {
        let config = MixedStreamConfig {
            batch_size: 20,
            seed: 0xDA7A_0005,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        };
        let total_steps = 17usize;
        let first_batch_index = 5u64;
        let mut prefetcher = MixedStreamBatchPrefetcher::with_queue_depth(
            config.clone(),
            total_steps,
            first_batch_index,
            4,
            3,
        )?;
        for expected_index in first_batch_index..first_batch_index + 3 {
            let (batch_index, prepared) = prefetcher.recv_next()?;
            assert_eq!(batch_index, expected_index);
            let (prefetched, prefetched_digest) = prepared.into_parts();
            let direct = compose_mixed_stream_batch(
                &config,
                expected_index as f32 / total_steps as f32,
                expected_index,
                V5DataSplit::Train,
            )?;
            let direct_digest =
                training_content_batch_digest(direct.transitions(), direct.content_masks())?;
            assert_eq!(
                prefetched
                    .factual()
                    .map(|factual| factual.pairwise_board_effect_labels()),
                direct
                    .factual()
                    .map(|factual| factual.pairwise_board_effect_labels())
            );
            assert_eq!(prefetched, direct);
            assert_eq!(prefetched_digest, direct_digest);
        }
        prefetcher.shutdown();
        assert!(prefetcher.workers.is_empty());
        assert!(!prefetcher.accepting);
        Ok(())
    }

    #[test]
    fn mixed_stream_prefetch_and_inline_content_chains_match() -> Result<()> {
        let config = MixedStreamConfig {
            batch_size: 20,
            seed: 0xDA7A_0006,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        };
        let total_steps = 6usize;
        let mut prefetcher =
            MixedStreamBatchPrefetcher::with_queue_depth(config.clone(), total_steps, 0, 3, 3)?;
        let mut prefetched_chain = [0; 32];
        let mut inline_chain = [0; 32];
        for expected_index in 0..total_steps as u64 {
            let (batch_index, prepared) = prefetcher.recv_next()?;
            assert_eq!(batch_index, expected_index);
            let (_, prefetched_digest) = prepared.into_parts();
            prefetched_chain = training_content_hash_append(prefetched_chain, prefetched_digest);

            let inline = compose_mixed_stream_batch(
                &config,
                expected_index as f32 / total_steps as f32,
                expected_index,
                V5DataSplit::Train,
            )?;
            let inline_digest =
                training_content_batch_digest(inline.transitions(), inline.content_masks())?;
            inline_chain = training_content_hash_append(inline_chain, inline_digest);
        }
        assert_eq!(prefetched_chain, inline_chain);
        prefetcher.shutdown();
        Ok(())
    }

    #[test]
    fn mixed_stream_generation_content_digest_snapshot() -> Result<()> {
        let config = MixedStreamConfig {
            batch_size: 20,
            seed: 0xDA7A_0008,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        };
        let total_steps = 3usize;
        let chain = (0..total_steps as u64).try_fold([0; 32], |chain, batch_index| {
            let batch = compose_mixed_stream_batch(
                &config,
                batch_index as f32 / total_steps as f32,
                batch_index,
                V5DataSplit::Train,
            )?;
            let digest = training_content_batch_digest(batch.transitions(), batch.content_masks())?;
            Ok::<_, anyhow::Error>(training_content_hash_append(chain, digest))
        })?;
        // Snapshot of the revision-4 stream: the objective binds the permuted
        // episode operator, so this value differs from the pre-conditioning
        // snapshot by design.
        assert_eq!(
            chain,
            [
                35, 79, 71, 109, 212, 65, 91, 45, 135, 210, 135, 44, 94, 92, 37, 86, 169, 16,
                234, 191, 235, 127, 60, 26, 55, 129, 23, 0, 113, 33, 225, 210,
            ]
        );
        Ok(())
    }
}
