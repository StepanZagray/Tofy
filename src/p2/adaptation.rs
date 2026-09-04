//! Test-time adaptation (ADR 0005 §6).
//!
//! Channel A ([`FactualBuffer::context_window`], exposed through
//! [`ContextProvider`]) is the bounded Context Window of factual transitions.
//! Channel B ([`FastWeightAdapter`]) is gradient adaptation of the Fast
//! Weights (§3.3 subset, [`crate::p2::model::FAST_WEIGHT_PREFIXES`]) on the
//! current game's own factual transitions. [`LiveContext`] is the Channel A
//! state a live policy owns for every `world_core_v6` checkpoint, with or
//! without Channel B (§6.1).
//!
//! Invariants enforced here:
//! - Predictions never enter the buffer: [`FactualBuffer::push`] is the only
//!   writer and the live driver calls it only for confirmed transitions.
//! - Only Fast Weights are ever written (`Var::set` on the named subset); the
//!   optimizer is constructed over that subset alone.
//! - The adaptation loss is the exact-decoder next-frame cross-entropy plus
//!   L2-SP. It carries no goal, terminal, Q or reliability term, and the
//!   fast subset excludes `event_head`, `q_head`, `reliability_head`,
//!   `goal_proj` and `prefix_head`, so goal/terminal readouts and the Phase A
//!   trust gates keep their pretrained parameters and calibration.
//! - Nothing in this module can write a checkpoint: it has no filesystem
//!   access, and [`FastWeightAdapter::restore_prior`] returns the model to
//!   theta_0 bitwise at the end of every game.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{ensure, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use clap::ValueEnum;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::p2::data::{ArcAction, ArcFrame, ContextTransition, CONTEXT_WINDOW_MAX};
use crate::p2::model::{
    is_fast_weight, unknown_operator_conditioning, ContextBatch, ContextBatchHost, RecursionDepth,
    RecursionOpts, WorldModel, PALETTE_SIZE,
};
use crate::p2::train::frames_to_indices;

/// AdamW learning rate on the fast subset (§6.2).
pub const ADAPT_LR: f64 = 1e-4;
/// L2-SP anchor weight `lambda * ||theta - theta_0||^2` (§6.2).
pub const ADAPT_L2SP_WEIGHT: f64 = 1e-3;
/// Global grad-norm clip over the fast subset (§6.2).
pub const ADAPT_GRAD_CLIP: f64 = 1.0;
/// Adaptation starts once the current level has this many unique transitions.
pub const ADAPT_MIN_LEVEL_TRANSITIONS: usize = 8;
/// At most this many gradient steps per newly observed transition.
pub const ADAPT_MAX_STEPS_PER_TRANSITION: usize = 4;
/// Cumulative steps per level `<= ADAPT_STEP_BUDGET_PER_UNIQUE * unique transitions`.
pub const ADAPT_STEP_BUDGET_PER_UNIQUE: usize = 8;
/// Batch size `min(ADAPT_BATCH_MAX, buffer len)`.
pub const ADAPT_BATCH_MAX: usize = 32;
/// Prequential guard window: the newest transitions scored before/after an update.
pub const ADAPT_PREQUENTIAL_WINDOW: usize = 4;
/// Collapse guard: skip an update whose grad-norm exceeds this multiple of the running mean.
pub const ADAPT_COLLAPSE_FACTOR: f64 = 3.0;
/// Deterministic reservoir-sampling seed (adaptation batches are reproducible per game).
const RESERVOIR_SEED: u64 = 0x0005_0602;

/// One confirmed transition from Factual Memory, tagged with its level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactualTransition {
    pub current: ArcFrame,
    pub action: ArcAction,
    pub next: ArcFrame,
    pub level_index: u16,
    /// Chronological position in the game (0-based, over appended entries).
    pub transition_index: usize,
    /// The Context Window that preceded this transition when it was observed
    /// (§6.2 "the row's own context"): `<= CONTEXT_WINDOW_MAX` earlier factual
    /// transitions of the buffer's scope, chronological. Frames are shared, so
    /// this is cheap to carry.
    pub context: Vec<ContextTransition>,
}

/// Which factual transitions feed the Context Window (§1.5, §6.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextScope {
    Level(u16),
    Game,
}

/// Live `--context-scope` (Channel A, §1.5/§6.1): the scope every decision's
/// window and every stored row context are drawn from. Independent of the
/// Channel B arm (`--adapt-carry` only decides whether fast weights reset at a
/// level boundary). The default is `game` because the training rows' windows
/// (Learning Histories, §1.5) cross level boundaries within an episode; a
/// level-scoped live window would be a distribution the model never saw at
/// exactly the moment it needs the rule evidence most (the first decisions of
/// a new level). Game boundaries always clear Factual Memory (§6.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum ContextScopeKind {
    /// Only the current level's factual transitions.
    Level,
    /// Every factual transition of the current game (training distribution).
    #[default]
    Game,
}

impl ContextScopeKind {
    /// The concrete scope for a decision or row at `level_index`.
    pub fn resolve(self, level_index: u16) -> ContextScope {
        match self {
            Self::Level => ContextScope::Level(level_index),
            Self::Game => ContextScope::Game,
        }
    }
}

/// Channel A source: supplies the last `<= CONTEXT_WINDOW_MAX` factual
/// transitions of the configured [`ContextScopeKind`] for every model call.
pub trait ContextProvider {
    fn context_window(&self) -> Vec<ContextTransition>;
}

/// Level-tagged, append-only store of factual transitions for one game.
/// Its [`ContextScopeKind`] decides the scope of every stored per-row context
/// and of every decision window: the row's own level or the whole game.
#[derive(Debug, Default)]
pub struct FactualBuffer {
    entries: Vec<FactualTransition>,
    seen: HashSet<(u16, u64, String)>,
    scope: ContextScopeKind,
}

fn raw_observation_identity(frame: &ArcFrame) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    frame.width.hash(&mut hasher);
    frame.height.hash(&mut hasher);
    frame.pixels.as_ref().hash(&mut hasher);
    hasher.finish()
}

fn action_key(action: &ArcAction) -> String {
    match (action.x, action.y) {
        (Some(x), Some(y)) => format!("{}:{x}:{y}", action.id),
        _ => action.id.to_string(),
    }
}

impl FactualBuffer {
    pub fn new(scope: ContextScopeKind) -> Self {
        Self {
            scope,
            ..Self::default()
        }
    }

    /// The configured `--context-scope`.
    pub fn scope(&self) -> ContextScopeKind {
        self.scope
    }

    /// Context scope for a decision or row at `level_index`.
    pub fn scope_for(&self, level_index: u16) -> ContextScope {
        self.scope.resolve(level_index)
    }

    /// Append a confirmed transition. Returns `false` (and stores nothing) when
    /// the same raw observation identity + action key was already seen in this
    /// level; a repeated factual query adds no new evidence. The entry carries
    /// the window that preceded it (never itself).
    pub fn push(
        &mut self,
        current: ArcFrame,
        action: ArcAction,
        next: ArcFrame,
        level_index: u16,
    ) -> bool {
        let key = (
            level_index,
            raw_observation_identity(&current),
            action_key(&action),
        );
        if !self.seen.insert(key) {
            return false;
        }
        let context = self.context_window(self.scope_for(level_index));
        self.entries.push(FactualTransition {
            current,
            action,
            next,
            level_index,
            transition_index: self.entries.len(),
            context,
        });
        true
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[FactualTransition] {
        &self.entries
    }

    pub fn unique_transitions_in_level(&self, level_index: u16) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.level_index == level_index)
            .count()
    }

    /// The `n` most recent entries in chronological order.
    pub fn newest(&self, n: usize) -> Vec<&FactualTransition> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].iter().collect()
    }

    /// Algorithm R reservoir sample of `min(k, len)` entries, without replacement.
    pub fn reservoir_sample<R: Rng>(&self, k: usize, rng: &mut R) -> Vec<&FactualTransition> {
        let mut reservoir: Vec<&FactualTransition> = Vec::with_capacity(k.min(self.entries.len()));
        for (index, entry) in self.entries.iter().enumerate() {
            if index < k {
                reservoir.push(entry);
            } else {
                let slot = rng.random_range(0..=index);
                if slot < k {
                    reservoir[slot] = entry;
                }
            }
        }
        reservoir
    }

    /// Channel A: the last `<= CONTEXT_WINDOW_MAX` factual transitions in the
    /// scope, chronological order.
    pub fn context_window(&self, scope: ContextScope) -> Vec<ContextTransition> {
        let mut window: Vec<ContextTransition> = self
            .entries
            .iter()
            .rev()
            .filter(|entry| match scope {
                ContextScope::Level(level) => entry.level_index == level,
                ContextScope::Game => true,
            })
            .take(CONTEXT_WINDOW_MAX)
            .map(|entry| ContextTransition {
                current: entry.current.clone(),
                action: entry.action.clone(),
                next: entry.next.clone(),
            })
            .collect();
        window.reverse();
        window
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.seen.clear();
    }
}

/// Channel A state owned by a live policy (ADR 0005 §6.1): the factual
/// transitions observed so far in the game, from which the window preceding
/// every decision is drawn. Disabled (`enabled = false`, the only legal state
/// for non-v6 checkpoints) it observes nothing and yields empty windows, so
/// every model call stays on the legacy `context = None` path.
#[derive(Debug, Default)]
pub struct LiveContext {
    buffer: FactualBuffer,
    enabled: bool,
}

impl LiveContext {
    pub fn new(enabled: bool, scope: ContextScopeKind) -> Self {
        Self {
            buffer: FactualBuffer::new(scope),
            enabled,
        }
    }

    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The configured `--context-scope` (reported even when the window is off).
    pub fn scope(&self) -> ContextScopeKind {
        self.buffer.scope()
    }

    /// New game: Factual Memory starts empty (§6.3).
    pub fn begin_game(&mut self) {
        self.buffer.clear();
    }

    /// Record a confirmed factual transition of `level_index`; call this only
    /// after the environment confirmed `next` (never for the pending action).
    pub fn observe(
        &mut self,
        current: &ArcFrame,
        action: &ArcAction,
        next: &ArcFrame,
        level_index: u16,
    ) {
        if self.enabled {
            self.buffer
                .push(current.clone(), action.clone(), next.clone(), level_index);
        }
    }

    /// The window for a decision taken at `level_index`: the most recent
    /// `<= CONTEXT_WINDOW_MAX` factual transitions of the configured scope,
    /// all observed strictly before this decision.
    pub fn window(&self, level_index: u16) -> Vec<ContextTransition> {
        if !self.enabled {
            return Vec::new();
        }
        self.buffer
            .context_window(self.buffer.scope_for(level_index))
    }
}

/// Device batch of one window replicated for `rows` candidates; `None` for an
/// empty window (the model then computes `c = 0` exactly, §3.1).
pub fn context_batch_for(
    window: &[ContextTransition],
    rows: usize,
    device: &Device,
) -> Result<Option<ContextBatch>> {
    ContextBatchHost::broadcast(window, rows)?
        .map(|host| ContextBatch::from_host(&host, device))
        .transpose()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptationMode {
    /// Fast weights reset to theta_0 at every level boundary (default arm).
    #[default]
    Reset,
    /// Preregistered arm: fast weights persist across levels within a game.
    Carry,
}

/// Per-decision Channel B telemetry (§6.2, recorded in `ActionDecision`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AdaptationTrace {
    pub updates: usize,
    pub skipped: usize,
    pub reverted: usize,
    pub steps_this_level: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preq_loss_before: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preq_loss_after: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    pub mode: AdaptationMode,
    /// Why no gradient step ran (`warmup`, `level_step_cap`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

struct FastVar {
    name: String,
    var: Var,
    theta0: Tensor,
    best: Tensor,
}

/// Shared handle to theta_0 of the fast subset (§6.2 prior-weight readouts).
///
/// Goal/terminal/reliability readouts consumed by Phase A trust and by the
/// greedy scorer must come from the prior weights, not from latents produced
/// by adapted dynamics. [`PriorWeights::with_prior_weights`] swaps the fast
/// subset to theta_0 (bitwise) for the duration of one readout call and
/// restores the adapted values afterwards, also on error. The handle is
/// `Clone` (Arc-shared with the adapter): the `Var`s share storage with the
/// model, theta_0 is cached on device at construction, and the adapter flags
/// drift after any optimizer step so an un-drifted model skips the swap.
#[derive(Clone)]
pub struct PriorWeights {
    inner: Arc<PriorWeightsInner>,
}

struct PriorWeightsInner {
    /// `(name, live Var shared with the model, theta_0 on device)`.
    fast: Vec<(String, Var, Tensor)>,
    /// Set after any optimizer step; cleared by `reset_to_prior`.
    drifted: AtomicBool,
}

impl PriorWeights {
    fn new(fast: &[FastVar]) -> Self {
        Self {
            inner: Arc::new(PriorWeightsInner {
                fast: fast
                    .iter()
                    .map(|entry| (entry.name.clone(), entry.var.clone(), entry.theta0.clone()))
                    .collect(),
                drifted: AtomicBool::new(false),
            }),
        }
    }

    fn set_drifted(&self, drifted: bool) {
        self.inner.drifted.store(drifted, Ordering::Release);
    }

    /// `true` while no optimizer step has moved the fast subset since the
    /// last reset, i.e. the live weights are theta_0 and no swap is needed.
    pub fn is_at_prior(&self) -> bool {
        !self.inner.drifted.load(Ordering::Acquire)
    }

    pub fn fast_weight_names(&self) -> Vec<&str> {
        self.inner
            .fast
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect()
    }

    /// Run `f` with the fast subset set to theta_0 bitwise, then restore the
    /// adapted values (whether or not `f` failed). A model that has not
    /// drifted runs `f` directly.
    pub fn with_prior_weights<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.is_at_prior() {
            return f();
        }
        let mut saved = Vec::with_capacity(self.inner.fast.len());
        for (_, var, theta0) in &self.inner.fast {
            saved.push(snapshot(var)?);
            var.set(theta0)?;
        }
        let result = f();
        let mut restore_error = None;
        for ((_, var, _), adapted) in self.inner.fast.iter().zip(&saved) {
            if let Err(err) = var.set(adapted) {
                restore_error.get_or_insert(err);
            }
        }
        if let Some(err) = restore_error {
            return Err(anyhow::anyhow!(
                "restoring adapted fast weights after a prior-weight readout failed: {err}"
            ));
        }
        result
    }
}

/// Channel B: gradient adaptation of the Fast Weights on factual transitions.
pub struct FastWeightAdapter<'a> {
    model: &'a WorldModel,
    device: &'a Device,
    fast: Vec<FastVar>,
    prior: PriorWeights,
    optimizer: AdamW,
    buffer: FactualBuffer,
    mode: AdaptationMode,
    level: u16,
    steps_this_level: usize,
    pending: usize,
    consecutive_worsenings: usize,
    grad_norm_sum: f64,
    grad_norm_count: usize,
    l2sp_weight: f64,
    min_level_transitions: usize,
    rng: StdRng,
}

fn snapshot(var: &Var) -> Result<Tensor> {
    Ok(var.as_tensor().detach().copy()?)
}

fn bitwise_equal(a: &Tensor, b: &Tensor) -> Result<bool> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() {
        return Ok(false);
    }
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()))
}

fn new_optimizer(fast: &[FastVar]) -> Result<AdamW> {
    AdamW::new(
        fast.iter().map(|entry| entry.var.clone()).collect(),
        ParamsAdamW {
            lr: ADAPT_LR,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )
    .map_err(Into::into)
}

impl<'a> FastWeightAdapter<'a> {
    /// Snapshot theta_0 of the fast subset from `varmap` (the VarMap `model`
    /// was built from) and build the fast-only AdamW. `scope` is the
    /// `--context-scope` every stored row's own context is drawn from (§6.2
    /// "the row's own context"); `mode` only decides the level-boundary reset.
    pub fn new(
        model: &'a WorldModel,
        varmap: &VarMap,
        device: &'a Device,
        mode: AdaptationMode,
        scope: ContextScopeKind,
    ) -> Result<Self> {
        ensure!(
            model.config().world_core_v4,
            "test-time adaptation requires the exact decoder (world_core_v4)"
        );
        let mut fast = Vec::new();
        {
            let data = varmap.data().lock().unwrap();
            for (name, var) in data.iter().filter(|(name, _)| is_fast_weight(name)) {
                let theta0 = snapshot(var)?;
                fast.push(FastVar {
                    name: name.clone(),
                    var: var.clone(),
                    best: theta0.clone(),
                    theta0,
                });
            }
        }
        fast.sort_by(|a, b| a.name.cmp(&b.name));
        ensure!(
            !fast.is_empty(),
            "no fast-weight parameters found in VarMap"
        );
        let optimizer = new_optimizer(&fast)?;
        let prior = PriorWeights::new(&fast);
        Ok(Self {
            model,
            device,
            fast,
            prior,
            optimizer,
            buffer: FactualBuffer::new(scope),
            mode,
            level: 0,
            steps_this_level: 0,
            pending: 0,
            consecutive_worsenings: 0,
            grad_norm_sum: 0.0,
            grad_norm_count: 0,
            l2sp_weight: ADAPT_L2SP_WEIGHT,
            min_level_transitions: ADAPT_MIN_LEVEL_TRANSITIONS,
            rng: StdRng::seed_from_u64(RESERVOIR_SEED),
        })
    }

    pub fn mode(&self) -> AdaptationMode {
        self.mode
    }

    pub fn buffer(&self) -> &FactualBuffer {
        &self.buffer
    }

    pub fn fast_weight_names(&self) -> Vec<&str> {
        self.fast.iter().map(|entry| entry.name.as_str()).collect()
    }

    /// Shared theta_0 handle for the policies' prior-weight readouts (§6.2).
    pub fn prior_weights(&self) -> PriorWeights {
        self.prior.clone()
    }

    /// Run `f(model)` with the fast subset at theta_0 bitwise; the adapted
    /// values are restored afterwards. See [`PriorWeights::with_prior_weights`].
    pub fn with_prior_weights<T>(&self, f: impl FnOnce(&WorldModel) -> Result<T>) -> Result<T> {
        let model = self.model;
        self.prior.with_prior_weights(|| f(model))
    }

    /// Falsifier knob for the L2-SP ablation arm; production keeps the default.
    pub fn set_l2sp_weight(&mut self, weight: f64) {
        self.l2sp_weight = weight;
    }

    /// Falsifier knob: the per-level warm-up (§6.2 default
    /// [`ADAPT_MIN_LEVEL_TRANSITIONS`]). Synthetic Learning Histories carry
    /// only `LEARNING_HISTORY_STEPS_PER_LEVEL + 1 = 7` transitions per level,
    /// so the §5.2 falsifier cannot leave warm-up at the default; a lowered
    /// value is a recorded deviation, never the live default.
    pub fn set_min_level_transitions(&mut self, transitions: usize) {
        self.min_level_transitions = transitions.max(1);
    }

    pub fn min_level_transitions(&self) -> usize {
        self.min_level_transitions
    }

    /// Reseed the reservoir sampler (deterministic per-episode batches for the
    /// falsifier; the live loop keeps the fixed `RESERVOIR_SEED`).
    pub fn reseed_reservoir(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Record one confirmed factual transition of `level_index`. Every observed
    /// transition (unique or not) arms up to `ADAPT_MAX_STEPS_PER_TRANSITION`
    /// steps; only unique ones extend the per-level step budget.
    pub fn observe(
        &mut self,
        current: &ArcFrame,
        action: &ArcAction,
        next: &ArcFrame,
        level_index: u16,
    ) -> bool {
        self.level = level_index;
        self.pending += 1;
        self.buffer
            .push(current.clone(), action.clone(), next.clone(), level_index)
    }

    /// Start a fresh game: prior weights, empty buffer.
    pub fn begin_game(&mut self) -> Result<()> {
        self.restore_prior()
    }

    /// Level boundary: the step budget restarts; the default arm also resets
    /// the fast weights to theta_0.
    pub fn on_level_transition(&mut self, level_index: u16) -> Result<()> {
        self.level = level_index;
        self.steps_this_level = 0;
        match self.mode {
            AdaptationMode::Reset => self.reset_to_prior(),
            AdaptationMode::Carry => {
                self.consecutive_worsenings = 0;
                Ok(())
            }
        }
    }

    /// Fast weights <- theta_0; theta_best <- theta_0; fresh optimizer state.
    pub fn reset_to_prior(&mut self) -> Result<()> {
        for entry in &mut self.fast {
            entry.var.set(&entry.theta0)?;
            entry.best = entry.theta0.clone();
        }
        self.optimizer = new_optimizer(&self.fast)?;
        self.prior.set_drifted(false);
        self.steps_this_level = 0;
        self.consecutive_worsenings = 0;
        self.grad_norm_sum = 0.0;
        self.grad_norm_count = 0;
        Ok(())
    }

    /// Game end: discard adapted weights and the game's buffer (§6.3). Fails
    /// closed if the model does not equal theta_0 bitwise afterwards.
    pub fn restore_prior(&mut self) -> Result<()> {
        self.reset_to_prior()?;
        self.buffer.clear();
        self.pending = 0;
        ensure!(
            self.fast_weights_equal_prior()?,
            "fast weights differ from theta_0 after restore_prior"
        );
        Ok(())
    }

    pub fn fast_weights_equal_prior(&self) -> Result<bool> {
        for entry in &self.fast {
            if !bitwise_equal(entry.var.as_tensor(), &entry.theta0)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `||theta - theta_0||^2` over the fast subset.
    pub fn drift_from_prior(&self) -> Result<f64> {
        let mut total = 0.0f64;
        for entry in &self.fast {
            total += f64::from(
                entry
                    .var
                    .as_tensor()
                    .detach()
                    .sub(&entry.theta0)?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
        }
        Ok(total)
    }

    /// §6.2 update rule. Returns `None` when nothing was observed since the
    /// last call; otherwise every decision (update, skip, revert) is traced.
    pub fn maybe_update(&mut self) -> Result<Option<AdaptationTrace>> {
        if self.pending == 0 {
            return Ok(None);
        }
        let pending = std::mem::take(&mut self.pending);
        let mut trace = AdaptationTrace {
            mode: self.mode,
            steps_this_level: self.steps_this_level,
            ..AdaptationTrace::default()
        };
        let unique = self.buffer.unique_transitions_in_level(self.level);
        if unique < self.min_level_transitions {
            trace.note = Some("warmup".into());
            return Ok(Some(trace));
        }
        let budget = (ADAPT_STEP_BUDGET_PER_UNIQUE * unique).saturating_sub(self.steps_this_level);
        let max_steps = (ADAPT_MAX_STEPS_PER_TRANSITION * pending).min(budget);
        if max_steps == 0 {
            trace.skipped = 1;
            trace.note = Some("level_step_cap".into());
            return Ok(Some(trace));
        }

        let preq_before = self.prequential_loss()?;
        trace.preq_loss_before = Some(preq_before);
        for _ in 0..max_steps {
            let batch = self.buffer.reservoir_sample(ADAPT_BATCH_MAX, &mut self.rng);
            let loss = self.next_frame_loss(&batch)?.add(&self.l2sp_penalty()?)?;
            let mut grads = loss.backward()?;
            let norm = self.fast_grad_norm(&grads)?;
            trace.grad_norm = Some(norm);
            let running_mean = (self.grad_norm_count > 0)
                .then(|| self.grad_norm_sum / self.grad_norm_count as f64);
            if !norm.is_finite()
                || running_mean.is_some_and(|mean| norm > ADAPT_COLLAPSE_FACTOR * mean)
            {
                trace.skipped += 1;
                continue;
            }
            if norm > ADAPT_GRAD_CLIP {
                self.scale_fast_grads(&mut grads, ADAPT_GRAD_CLIP / norm)?;
            }
            self.optimizer.step(&grads)?;
            self.prior.set_drifted(true);
            self.grad_norm_sum += norm;
            self.grad_norm_count += 1;
            self.steps_this_level += 1;
            trace.updates += 1;
        }
        trace.steps_this_level = self.steps_this_level;
        if trace.updates > 0 {
            let preq_after = self.prequential_loss()?;
            trace.preq_loss_after = Some(preq_after);
            self.prequential_guard(preq_before, preq_after, &mut trace)?;
        }
        Ok(Some(trace))
    }

    /// Two consecutive worsenings of the prequential loss revert to theta_best;
    /// a non-worsening update promotes the current weights to theta_best.
    pub(crate) fn prequential_guard(
        &mut self,
        before: f64,
        after: f64,
        trace: &mut AdaptationTrace,
    ) -> Result<()> {
        if after > before {
            self.consecutive_worsenings += 1;
        } else {
            self.consecutive_worsenings = 0;
            for entry in &mut self.fast {
                entry.best = snapshot(&entry.var)?;
            }
        }
        if self.consecutive_worsenings >= 2 {
            for entry in &self.fast {
                entry.var.set(&entry.best)?;
            }
            self.optimizer = new_optimizer(&self.fast)?;
            self.consecutive_worsenings = 0;
            trace.reverted += 1;
        }
        Ok(())
    }

    #[cfg(test)]
    fn fast_weights_equal_best(&self) -> Result<bool> {
        for entry in &self.fast {
            if !bitwise_equal(entry.var.as_tensor(), &entry.best)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn prequential_loss(&self) -> Result<f64> {
        let newest = self.buffer.newest(ADAPT_PREQUENTIAL_WINDOW);
        Ok(f64::from(
            self.next_frame_loss(&newest)?.detach().to_scalar::<f32>()?,
        ))
    }

    /// ADR 0003 exact-decoder next-frame loss on factual rows: unimix
    /// cross-entropy of the decoded next frame against the observed next frame,
    /// mean over every decoded pixel and row. The exact decoder's own row count
    /// governs (legacy heads decode 63 rows; `world_core_v6` decodes all 64).
    /// On `world_core_v6` every row is conditioned on its own stored context
    /// (§6.2); legacy checkpoints get `None`.
    fn next_frame_loss(&self, rows: &[&FactualTransition]) -> Result<Tensor> {
        ensure!(
            !rows.is_empty(),
            "adaptation loss requires at least one row"
        );
        let n = rows.len();
        let currents: Vec<ArcFrame> = rows.iter().map(|row| row.current.clone()).collect();
        let nexts: Vec<ArcFrame> = rows.iter().map(|row| row.next.clone()).collect();
        let frames = frames_to_indices(&currents, self.device)?;
        let next_frames = frames_to_indices(&nexts, self.device)?;
        let actions = Tensor::from_vec(
            rows.iter()
                .map(|row| u32::from(row.action.id))
                .collect::<Vec<_>>(),
            n,
            self.device,
        )?;
        let coords = Tensor::from_vec(
            rows.iter()
                .flat_map(|row| {
                    [
                        row.action.x.map_or(0.0, |x| f32::from(x) / 63.0),
                        row.action.y.map_or(0.0, |y| f32::from(y) / 63.0),
                    ]
                })
                .collect::<Vec<_>>(),
            (n, 2),
            self.device,
        )?;
        let operator = unknown_operator_conditioning(n, self.device)?;
        let context = if self.model.config().world_core_v6 {
            ContextBatchHost::from_windows(rows.iter().map(|row| row.context.iter()))?
                .map(|host| ContextBatch::from_host(&host, self.device))
                .transpose()?
        } else {
            None
        };
        let encoded = self
            .model
            .encode_state_pair_for_training(&frames, &next_frames)?;
        let canonical = self.model.canonical_representation(&encoded.current)?;
        let out = self
            .model
            .full_v4_training_latents_from_encoded_state_with_operator_conditioning_with_context(
                &encoded.current,
                &canonical,
                &actions,
                &coords,
                &operator,
                context.as_ref(),
                RecursionDepth::from_config(self.model.config()),
                0.0,
                None,
                RecursionOpts::training(true),
            )?;
        let logits = self.model.exact_gameplay_logits_trainable(&out.y)?;
        let decoded_rows = logits.dim(1)?;
        let labels = next_frames
            .narrow(2, 0, decoded_rows)?
            .squeeze(1)?
            .to_dtype(DType::U32)?
            .contiguous()?;
        let pixels = labels.elem_count();
        let selected = candle_nn::ops::softmax(&logits, D::Minus1)?
            .reshape((pixels, PALETTE_SIZE))?
            .gather(&labels.flatten_all()?.unsqueeze(1)?, 1)?;
        selected
            .affine(0.99, 0.01 / PALETTE_SIZE as f64)?
            .log()?
            .neg()?
            .mean_all()
            .map_err(Into::into)
    }

    fn l2sp_penalty(&self) -> Result<Tensor> {
        let mut total = Tensor::zeros((), DType::F32, self.device)?;
        for entry in &self.fast {
            total = total.add(
                &entry
                    .var
                    .as_tensor()
                    .sub(&entry.theta0)?
                    .sqr()?
                    .sum_all()?
                    .to_dtype(DType::F32)?,
            )?;
        }
        total.affine(self.l2sp_weight, 0.0).map_err(Into::into)
    }

    fn fast_grad_norm(&self, grads: &GradStore) -> Result<f64> {
        let mut sum_sq = 0.0f64;
        for entry in &self.fast {
            if let Some(grad) = grads.get(entry.var.as_tensor()) {
                sum_sq += f64::from(
                    grad.to_dtype(DType::F32)?
                        .sqr()?
                        .sum_all()?
                        .to_scalar::<f32>()?,
                );
            }
        }
        Ok(sum_sq.sqrt())
    }

    fn scale_fast_grads(&self, grads: &mut GradStore, scale: f64) -> Result<()> {
        for entry in &self.fast {
            let tensor = entry.var.as_tensor();
            if let Some(grad) = grads.get(tensor) {
                let scaled = grad.affine(scale, 0.0)?;
                grads.insert(tensor, scaled);
            }
        }
        Ok(())
    }
}

impl ContextProvider for FastWeightAdapter<'_> {
    fn context_window(&self) -> Vec<ContextTransition> {
        self.buffer
            .context_window(self.buffer.scope_for(self.level))
    }
}

impl std::fmt::Debug for FastWeightAdapter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastWeightAdapter")
            .field("mode", &self.mode)
            .field("level", &self.level)
            .field("fast_weights", &self.fast.len())
            .field("buffer_len", &self.buffer.len())
            .field("steps_this_level", &self.steps_this_level)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::data::FRAME_SIDE;
    use crate::p2::experiment::ConsumerReadoutTopology;
    use crate::p2::model::ModelConfig;
    use candle_nn::VarBuilder;

    fn frame(seed: u8) -> ArcFrame {
        let pixels = (0..FRAME_SIDE * FRAME_SIDE)
            .map(|index| ((index as u32 * 7 + u32::from(seed) * 13) % 16) as u8)
            .collect();
        ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels).unwrap()
    }

    fn action(id: u8) -> ArcAction {
        ArcAction::new(id, None, None).unwrap()
    }

    fn tiny_model(device: &Device) -> Result<(WorldModel, VarMap)> {
        tiny_model_with(device, false)
    }

    fn tiny_model_with(device: &Device, world_core_v6: bool) -> Result<(WorldModel, VarMap)> {
        let cfg = ModelConfig {
            patch_size: 4,
            hidden_dim: 8,
            action_dim: 8,
            goal_dim: 6,
            inner_steps: 1,
            outer_steps: 1,
            spatial_action_field: true,
            world_core_v4: true,
            world_core_v5: true,
            world_core_v6,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, device))?;
        Ok((model, varmap))
    }

    /// The context FiLM is zero-initialised (v5 recovered exactly), so make
    /// the channel observable before asserting that a window reached the model.
    fn perturb_context_film(varmap: &VarMap) -> Result<()> {
        let data = varmap.data().lock().unwrap();
        let var = data
            .get("context_film_gamma.weight")
            .expect("world_core_v6 context FiLM parameter");
        var.set(&var.as_tensor().ones_like()?.affine(0.5, 0.0)?)?;
        Ok(())
    }

    fn all_var_snapshots(varmap: &VarMap) -> Result<Vec<(String, Tensor)>> {
        let data = varmap.data().lock().unwrap();
        let mut out = Vec::new();
        for (name, var) in data.iter() {
            out.push((name.clone(), snapshot(var)?));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn observe_distinct(adapter: &mut FastWeightAdapter<'_>, count: u8, level: u16) {
        for i in 0..count {
            let appended = adapter.observe(
                &frame(i),
                &action(1 + i % 4),
                &frame(i.wrapping_add(1)),
                level,
            );
            assert!(appended);
        }
    }

    #[test]
    fn buffer_is_chronological_dedups_and_bounds_windows() {
        let mut buffer = FactualBuffer::default();
        for i in 0..20u8 {
            assert!(buffer.push(frame(i), action(1), frame(i + 1), i as u16 / 10));
        }
        // Duplicate (state, action) within the same level is not appended.
        assert!(!buffer.push(frame(3), action(1), frame(9), 0));
        // Same state, different action is a new transition.
        assert!(buffer.push(frame(3), action(2), frame(9), 0));
        assert_eq!(buffer.len(), 21);
        assert_eq!(buffer.unique_transitions_in_level(0), 11);
        assert_eq!(buffer.unique_transitions_in_level(1), 10);
        let indices: Vec<_> = buffer
            .entries()
            .iter()
            .map(|entry| entry.transition_index)
            .collect();
        assert_eq!(indices, (0..21).collect::<Vec<_>>());

        let window = buffer.context_window(ContextScope::Game);
        assert_eq!(window.len(), CONTEXT_WINDOW_MAX);
        let expected: Vec<_> = buffer.entries()[21 - CONTEXT_WINDOW_MAX..]
            .iter()
            .map(|entry| entry.current.clone())
            .collect();
        let got: Vec<_> = window.iter().map(|entry| entry.current.clone()).collect();
        assert_eq!(got, expected, "chronological order, newest last");
        let level1 = buffer.context_window(ContextScope::Level(1));
        assert_eq!(level1.len(), 10);
        assert_eq!(level1[0].current, frame(10));
        assert_eq!(level1[9].current, frame(19));

        let mut rng = StdRng::seed_from_u64(1);
        let sample = buffer.reservoir_sample(ADAPT_BATCH_MAX, &mut rng);
        assert_eq!(sample.len(), 21.min(ADAPT_BATCH_MAX));
        let mut distinct: Vec<_> = sample.iter().map(|e| e.transition_index).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), sample.len(), "without replacement");
        let small = buffer.reservoir_sample(5, &mut rng);
        assert_eq!(small.len(), 5);
        assert_eq!(FactualBuffer::default().newest(4).len(), 0);
        assert_eq!(buffer.newest(4).len(), 4);
        assert_eq!(buffer.newest(4)[3].transition_index, 20);
    }

    #[test]
    fn buffer_entries_carry_their_preceding_window() {
        // Level scope: each row's context is the earlier rows of its own level.
        let mut buffer = FactualBuffer::new(ContextScopeKind::Level);
        for i in 0..20u8 {
            assert!(buffer.push(frame(i), action(1), frame(i + 1), i as u16 / 10));
        }
        let entries = buffer.entries();
        assert!(entries[0].context.is_empty());
        let firsts: Vec<_> = entries[5]
            .context
            .iter()
            .map(|c| c.current.clone())
            .collect();
        assert_eq!(firsts, (0..5).map(frame).collect::<Vec<_>>());
        assert!(
            entries[10].context.is_empty(),
            "level 1 starts with no context"
        );
        assert_eq!(entries[19].context.len(), 9);
        assert_eq!(entries[19].context[0].current, frame(10));
        for entry in entries {
            assert!(
                entry.context.iter().all(|c| c.current != entry.current),
                "a row never carries itself"
            );
        }
        // Game scope (the default): bounded by CONTEXT_WINDOW_MAX.
        let mut carry = FactualBuffer::new(ContextScopeKind::Game);
        assert_eq!(FactualBuffer::default().scope(), ContextScopeKind::Game);
        for i in 0..20u8 {
            assert!(carry.push(frame(i), action(1), frame(i + 1), i as u16 / 10));
        }
        assert_eq!(carry.entries()[10].context.len(), 10);
        assert_eq!(carry.entries()[19].context.len(), CONTEXT_WINDOW_MAX);
        assert_eq!(carry.entries()[19].context[0].current, frame(3));
        assert_eq!(carry.entries()[19].context[15].current, frame(18));
    }

    #[test]
    fn live_context_window_precedes_the_decision_and_respects_the_flag() {
        let mut off = LiveContext::disabled();
        off.observe(&frame(0), &action(1), &frame(1), 0);
        assert!(!off.enabled());
        assert!(off.window(0).is_empty());

        let mut live = LiveContext::new(true, ContextScopeKind::Level);
        assert!(live.enabled());
        for i in 0..20u8 {
            // The window handed to decision `i` never contains transition `i`.
            let window = live.window(i as u16 / 10);
            assert!(window.iter().all(|c| c.current != frame(i)));
            assert_eq!(window.len(), (i as usize % 10).min(CONTEXT_WINDOW_MAX));
            live.observe(&frame(i), &action(1), &frame(i + 1), i as u16 / 10);
        }
        assert_eq!(live.window(1).len(), 10);
        assert!(live.window(2).is_empty(), "a new level starts empty");
        live.begin_game();
        assert!(live.window(1).is_empty());

        let mut carry = LiveContext::new(true, ContextScopeKind::Game);
        for i in 0..20u8 {
            carry.observe(&frame(i), &action(1), &frame(i + 1), i as u16 / 10);
        }
        assert_eq!(carry.window(2).len(), CONTEXT_WINDOW_MAX);
        assert_eq!(carry.window(2)[15].current, frame(19));
        // Game boundary invariant: Factual Memory is emptied whatever the scope.
        carry.begin_game();
        assert!(carry.window(2).is_empty());
        assert!(carry.window(0).is_empty());
        assert_eq!(carry.scope(), ContextScopeKind::Game);
    }

    #[test]
    fn context_scope_kind_is_independent_of_the_adaptation_arm() -> Result<()> {
        // Channel A scope is a first-class knob: a reset-arm adapter may draw
        // game-scoped row contexts and a carry-arm adapter level-scoped ones.
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let mut reset_game = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Game,
        )?;
        observe_distinct(&mut reset_game, 5, 0);
        reset_game.on_level_transition(1)?;
        assert_eq!(reset_game.mode(), AdaptationMode::Reset);
        assert_eq!(reset_game.buffer().scope(), ContextScopeKind::Game);
        assert_eq!(reset_game.context_window().len(), 5);
        assert_eq!(reset_game.buffer().scope_for(1), ContextScope::Game);

        let mut carry_level = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Carry,
            ContextScopeKind::Level,
        )?;
        observe_distinct(&mut carry_level, 5, 0);
        carry_level.on_level_transition(1)?;
        assert_eq!(carry_level.mode(), AdaptationMode::Carry);
        assert!(carry_level.context_window().is_empty());
        assert_eq!(carry_level.buffer().scope_for(1), ContextScope::Level(1));
        assert_eq!(ContextScopeKind::default(), ContextScopeKind::Game);
        assert_eq!(
            serde_json::to_value(ContextScopeKind::Level)?,
            serde_json::Value::String("level".into())
        );
        Ok(())
    }

    #[test]
    fn v6_next_frame_loss_conditions_each_row_on_its_own_context() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model_with(&device, true)?;
        perturb_context_film(&varmap)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        assert!(adapter
            .fast_weight_names()
            .contains(&"context_film_gamma.weight"));
        observe_distinct(&mut adapter, 4, 0);
        let rows = adapter.buffer.newest(4);
        assert_eq!(rows[3].context.len(), 3);
        let with = adapter.next_frame_loss(&rows)?.to_scalar::<f32>()?;
        let stripped: Vec<FactualTransition> = rows
            .iter()
            .map(|row| FactualTransition {
                context: Vec::new(),
                ..(*row).clone()
            })
            .collect();
        let refs: Vec<&FactualTransition> = stripped.iter().collect();
        let without = adapter.next_frame_loss(&refs)?.to_scalar::<f32>()?;
        assert!(with.is_finite() && without.is_finite());
        assert_ne!(with, without, "the v6 loss must see the rows' own context");
        // A full Channel B update runs with per-row context on v6.
        observe_distinct(&mut adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 1);
        adapter.on_level_transition(1)?;
        let trace = adapter.maybe_update()?.expect("pending");
        assert!(trace.updates > 0, "{trace:?}");
        adapter.restore_prior()?;

        // Legacy (v5) checkpoints: rows still carry a window, none is passed.
        let (v5, v5_map) = tiny_model(&device)?;
        assert!(!v5.config().world_core_v6);
        let mut legacy = FastWeightAdapter::new(
            &v5,
            &v5_map,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        observe_distinct(&mut legacy, 4, 0);
        let rows = legacy.buffer.newest(4);
        assert_eq!(rows[3].context.len(), 3);
        assert!(legacy
            .next_frame_loss(&rows)?
            .to_scalar::<f32>()?
            .is_finite());
        Ok(())
    }

    #[test]
    fn fast_subset_matches_real_parameter_names_and_excludes_goal_heads() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        let names = adapter.fast_weight_names();
        assert!(names.contains(&"pixel_emb.weight"), "{names:?}");
        assert!(names.contains(&"encoder.patch.weight"), "{names:?}");
        assert!(names.contains(&"action_film_gamma.weight"), "{names:?}");
        assert!(names.contains(&"action_film_beta.bias"), "{names:?}");
        assert!(names
            .iter()
            .any(|name| name.starts_with("exact_grounding_head.")));
        for frozen in [
            "event_head.weight",
            "q_head.weight",
            "reliability_head.weight",
            "goal_proj.weight",
            "prefix_head.weight",
            "encoder.c2.weight",
            "encoder.proj.weight",
            "block.c1.weight",
        ] {
            assert!(!is_fast_weight(frozen), "{frozen} must stay frozen");
        }
        Ok(())
    }

    #[test]
    fn updates_touch_only_fast_weights_and_restore_prior_is_bitwise() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let before = all_var_snapshots(&varmap)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        observe_distinct(&mut adapter, 7, 0);
        let trace = adapter.maybe_update()?.expect("pending");
        assert_eq!(trace.updates, 0);
        assert_eq!(trace.note.as_deref(), Some("warmup"));
        assert!(adapter.maybe_update()?.is_none(), "nothing pending");

        let mut updates = 0;
        let mut round = 7u8;
        while updates < 10 {
            adapter.observe(&frame(round), &action(1), &frame(round + 1), 0);
            round += 1;
            let trace = adapter.maybe_update()?.expect("pending");
            assert_eq!(trace.mode, AdaptationMode::Reset);
            assert!(trace.preq_loss_before.is_some());
            assert!(trace.updates + trace.skipped <= ADAPT_MAX_STEPS_PER_TRANSITION);
            updates += trace.updates;
        }
        assert!(updates >= 10);

        let after = all_var_snapshots(&varmap)?;
        let mut fast_changed = 0;
        for ((name, was), (_, now)) in before.iter().zip(&after) {
            let same = bitwise_equal(was, now)?;
            if is_fast_weight(name) {
                if !same {
                    fast_changed += 1;
                }
            } else {
                assert!(same, "frozen parameter {name} changed");
            }
        }
        assert!(fast_changed > 0, "adaptation moved no fast weight");
        assert!(!adapter.fast_weights_equal_prior()?);
        assert!(adapter.drift_from_prior()? > 0.0);

        adapter.restore_prior()?;
        assert!(adapter.fast_weights_equal_prior()?);
        assert!(adapter.buffer().is_empty());
        let restored = all_var_snapshots(&varmap)?;
        for ((name, was), (_, now)) in before.iter().zip(&restored) {
            assert!(bitwise_equal(was, now)?, "{name} differs from theta_0");
        }
        Ok(())
    }

    /// §6.2 prior-weight readouts: inside `with_prior_weights` every fast
    /// parameter is theta_0 bitwise, the adapted values come back afterwards
    /// (also when the closure fails), and an un-drifted adapter skips the swap.
    #[test]
    fn prior_weight_swap_is_bitwise_and_restores_the_adapted_weights() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let theta0 = all_var_snapshots(&varmap)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Game,
        )?;
        let prior = adapter.prior_weights();
        assert!(prior.is_at_prior());
        assert_eq!(prior.fast_weight_names(), adapter.fast_weight_names());
        adapter.set_min_level_transitions(1);
        let mut updates = 0;
        let mut round = 0u8;
        while updates == 0 {
            adapter.observe(&frame(round), &action(1), &frame(round + 1), 0);
            round += 1;
            updates += adapter.maybe_update()?.expect("pending").updates;
        }
        assert!(!prior.is_at_prior());
        assert!(!adapter.fast_weights_equal_prior()?);
        let adapted = all_var_snapshots(&varmap)?;

        adapter.with_prior_weights(|_| {
            let inside = all_var_snapshots(&varmap)?;
            for ((name, was), (_, now)) in theta0.iter().zip(&inside) {
                assert!(
                    bitwise_equal(was, now)?,
                    "{name} is not theta_0 inside the swap"
                );
            }
            Ok(())
        })?;
        let after = all_var_snapshots(&varmap)?;
        for ((name, was), (_, now)) in adapted.iter().zip(&after) {
            assert!(
                bitwise_equal(was, now)?,
                "{name} not restored after the swap"
            );
        }
        assert!(!prior.is_at_prior(), "the swap does not clear drift");

        // A failing readout still restores the adapted weights.
        let failed: Result<()> = prior.with_prior_weights(|| anyhow::bail!("readout failed"));
        assert!(failed.is_err());
        let after_error = all_var_snapshots(&varmap)?;
        for ((name, was), (_, now)) in adapted.iter().zip(&after_error) {
            assert!(
                bitwise_equal(was, now)?,
                "{name} not restored after an error"
            );
        }

        // Reset clears drift; the handle then runs closures without swapping.
        adapter.reset_to_prior()?;
        assert!(prior.is_at_prior());
        assert!(adapter.fast_weights_equal_prior()?);
        Ok(())
    }

    #[test]
    fn prequential_guard_reverts_to_best_after_two_worsenings() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        let mut trace = AdaptationTrace::default();
        // Improvement promotes the current weights to theta_best.
        adapter.prequential_guard(1.0, 0.5, &mut trace)?;
        let best = all_var_snapshots(&varmap)?;
        // Perturb every fast weight, then rig two consecutive worsenings.
        for entry in &adapter.fast {
            entry.var.set(&entry.var.as_tensor().affine(1.0, 0.25)?)?;
        }
        assert!(!adapter.fast_weights_equal_best()?);
        adapter.prequential_guard(0.5, 0.7, &mut trace)?;
        assert_eq!(trace.reverted, 0, "one worsening is tolerated");
        assert!(!adapter.fast_weights_equal_best()?);
        adapter.prequential_guard(0.7, 0.9, &mut trace)?;
        assert_eq!(trace.reverted, 1);
        assert!(adapter.fast_weights_equal_best()?);
        let now = all_var_snapshots(&varmap)?;
        for ((name, was), (_, is)) in best.iter().zip(&now) {
            assert!(bitwise_equal(was, is)?, "{name} not restored to theta_best");
        }
        Ok(())
    }

    #[test]
    fn step_caps_are_enforced_per_level() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        observe_distinct(&mut adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 0);
        let trace = adapter.maybe_update()?.expect("pending");
        // Eight pending transitions arm at most 4 * 8 = 32 attempted steps.
        assert!(
            trace.updates + trace.skipped <= ADAPT_MAX_STEPS_PER_TRANSITION * 8,
            "{trace:?}"
        );
        let budget = ADAPT_STEP_BUDGET_PER_UNIQUE * ADAPT_MIN_LEVEL_TRANSITIONS;
        // Re-observing an already-seen transition adds no budget but arms steps.
        let mut capped = None;
        for _ in 0..(budget * 2) {
            let appended = adapter.observe(&frame(0), &action(1), &frame(1), 0);
            assert!(!appended);
            let trace = adapter.maybe_update()?.expect("pending");
            assert!(trace.steps_this_level <= budget, "{trace:?}");
            if trace.note.as_deref() == Some("level_step_cap") {
                assert_eq!(trace.updates, 0);
                assert_eq!(trace.skipped, 1);
                capped = Some(trace);
                break;
            }
        }
        let capped = capped.expect("cap never reached");
        assert_eq!(capped.steps_this_level, budget);

        // A level boundary restarts the budget (default arm resets weights too).
        adapter.on_level_transition(1)?;
        assert!(adapter.fast_weights_equal_prior()?);
        observe_distinct(&mut adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 1);
        let trace = adapter.maybe_update()?.expect("pending");
        assert!(trace.updates > 0, "{trace:?}");
        assert!(trace.steps_this_level < budget);
        Ok(())
    }

    #[test]
    fn carry_arm_keeps_weights_across_levels() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Carry,
            ContextScopeKind::Game,
        )?;
        observe_distinct(&mut adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 0);
        let trace = adapter.maybe_update()?.expect("pending");
        assert!(trace.updates > 0);
        assert_eq!(trace.mode, AdaptationMode::Carry);
        adapter.on_level_transition(1)?;
        assert!(!adapter.fast_weights_equal_prior()?, "carry keeps weights");
        assert_eq!(adapter.context_window().len(), ADAPT_MIN_LEVEL_TRANSITIONS);
        adapter.restore_prior()?;
        assert!(adapter.fast_weights_equal_prior()?);
        Ok(())
    }

    #[test]
    fn l2sp_reduces_drift_versus_no_penalty_arm() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = tiny_model(&device)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        let run = |adapter: &mut FastWeightAdapter<'_>, weight: f64| -> Result<f64> {
            adapter.restore_prior()?;
            adapter.rng = StdRng::seed_from_u64(RESERVOIR_SEED);
            adapter.set_l2sp_weight(weight);
            observe_distinct(adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 0);
            for round in 0..3u8 {
                adapter.observe(&frame(40 + round), &action(2), &frame(41 + round), 0);
                adapter.maybe_update()?;
            }
            adapter.drift_from_prior()
        };
        let anchored = run(&mut adapter, 1e3)?;
        let free = run(&mut adapter, 0.0)?;
        assert!(anchored > 0.0 && free > 0.0);
        assert!(
            anchored < free,
            "L2-SP arm drift {anchored} should be below free arm drift {free}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_update_touches_only_fast_weights_and_restores_prior() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("no CUDA device available; skipping");
            return Ok(());
        };
        let (model, varmap) = tiny_model(&device)?;
        let before = all_var_snapshots(&varmap)?;
        let mut adapter = FastWeightAdapter::new(
            &model,
            &varmap,
            &device,
            AdaptationMode::Reset,
            ContextScopeKind::Level,
        )?;
        observe_distinct(&mut adapter, ADAPT_MIN_LEVEL_TRANSITIONS as u8, 0);
        let trace = adapter.maybe_update()?.expect("pending");
        assert!(trace.updates > 0, "{trace:?}");
        assert!(trace.grad_norm.is_some_and(f64::is_finite));
        assert!(trace.preq_loss_after.is_some_and(f64::is_finite));
        let after = all_var_snapshots(&varmap)?;
        for ((name, was), (_, now)) in before.iter().zip(&after) {
            if !is_fast_weight(name) {
                assert!(bitwise_equal(was, now)?, "frozen {name} changed on CUDA");
            }
        }
        adapter.restore_prior()?;
        assert!(adapter.fast_weights_equal_prior()?);
        Ok(())
    }

    #[test]
    fn trace_serialization_skips_empty_fields() -> Result<()> {
        let trace = AdaptationTrace {
            updates: 1,
            mode: AdaptationMode::Carry,
            ..AdaptationTrace::default()
        };
        let json = serde_json::to_string(&trace)?;
        assert!(json.contains("\"mode\":\"carry\""));
        assert!(!json.contains("preq_loss_before"));
        assert!(!json.contains("grad_norm"));
        assert!(!json.contains("note"));
        let back: AdaptationTrace = serde_json::from_str(&json)?;
        assert_eq!(back, trace);
        Ok(())
    }
}
