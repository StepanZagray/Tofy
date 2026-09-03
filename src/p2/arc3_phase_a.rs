//! ADR 0004 Phase A: the rewire-only falsification controller.
//!
//! Adds no trained parameters. It puts an exact observed-state graph under the
//! frozen world model, keeps a tempered posterior over a bounded goal-candidate
//! set, screens actions at horizon 1, verifies a few finalists at horizon 2,
//! charges the selection-aware acceptance bound (Amendment 1), and executes
//! exactly one environment action before observing and replanning.
//!
//! Trust is calibration-driven and fails closed: without a calibration artifact
//! the controller reduces to the exact graph-frontier explorer and says so in
//! every decision trace. Kernels stay tensor-free behind [`PhaseAModel`].

use crate::p2::adaptation::{context_batch_for, LiveContext};
use crate::p2::arc3_live::{
    enumerate_actions_with, ActionDecision, ActionScore, ArcObservation, LivePolicy,
};
use crate::p2::data::{ArcAction, ArcFrame, ContextTransition, FRAME_SIDE, GOAL_FEATURES_DIM};
use crate::p2::latent_planning::adapter::{
    EvalBudget, GoalEventReadout, ModelCallError, PhaseAModel, StepPrediction,
};
use crate::p2::latent_planning::belief::BeliefState;
use crate::p2::latent_planning::config::PhaseAConfig;
use crate::p2::latent_planning::goals::{
    build_candidate_set, feature_inventory, CandidateSet, FeatureInventory, PlainFrame,
    FRAME_PIXELS,
};
use crate::p2::latent_planning::graph::{
    ActionKey, FactualEdge, ModelPredictionRecord, ObservedGraphError, ObservedStateGraph,
    RawObservationId, TerminalChannel,
};
use crate::p2::latent_planning::probe::{choose_probe, CandidateProbe, ProbeChoice, ProbeClaim};
use crate::p2::latent_planning::templates::{evaluate_predicate, propose_candidates};
use crate::p2::latent_planning::trust::{
    accept_finalist, selection_charge, EdgeTrust, PhaseACalibration,
};
use crate::p2::model::{
    unknown_operator_conditioning, RecursionDepth, RecursionOpts, WorldModel, EVENT_EXHAUSTED,
    EVENT_GOAL_FAILED, EVENT_GOAL_SATISFIED, EVENT_NOOP,
};
use crate::p2::train::frames_to_indices;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Instant;

pub const PHASE_A_POLICY: &str = "phase_a_falsification_v1";
pub const PHASE_A_POLICY_LIMITATION: &str = "ADR 0004 Phase A rewire-only controller: \
    exact observed-state graph, bounded goal-candidate posterior from a four-family \
    template vocabulary, horizon<=2 budgeted lookahead with calibration-driven \
    fail-closed trust gates and the 2*epsilon selection charge. No trained \
    parameters were added. Without a calibration artifact it reduces to graph-frontier \
    exploration. Irreversible-edge labeling, the inverse-action falsifier, the \
    exhausted channel, and the switch-order/preserve-resource families are deferred.";
pub const PHASE_A_GOAL_FEATURE_CONTRACT: &str = "g19 goal vectors are instantiated per \
    decision from the observed frame's feature inventory (reach_marker, collect_all, \
    avoid_hazard+reach_marker, trigger_terminal) and mirror GoalFeatures::encode; the \
    all-zero goal-dropout vector is always scored as the safety particle.";

/// Budget split per decision; must sum to at most `PhaseAConfig::max_model_evals`.
const ENCODE_EVALS: usize = 1;
const ROOT_SCREEN_EVALS: usize = 40;
const VERIFY_EVALS: usize = 14;
const DECODE_EVALS: usize = 8;
const MAX_FINALISTS: usize = 3;
const MAX_EXTENSIONS_PER_FINALIST: usize = 5;
const MAX_DECODED_PREFIXES: usize = 6;
/// Raw satisfied-probability threshold for a prefix to be considered a claimant
/// before calibration lowers it to a bound.
const CLAIM_RAW_THRESHOLD: f32 = 0.5;
/// Unknown-goal mass above which the goal-dropout particle joins the protected set.
const UNKNOWN_PROTECT_THRESHOLD: f64 = 0.05;

/// Per-decision telemetry attached to `ActionDecision.phase_a`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseADecisionTrace {
    pub mode: String,
    pub model_evals: usize,
    pub event_head_reads: usize,
    pub elapsed_ms: u64,
    pub goal_candidates: usize,
    pub unknown_mass: f64,
    pub protected: usize,
    pub claim_mass: f64,
    pub selection_charge: f64,
    pub deadline_truncated: bool,
    pub calibration_missing: bool,
    pub prefix: Vec<String>,
    /// Screening accounting: how many horizon-1 roots were scored and why they
    /// were dropped. Without these a frontier fallback is unexplainable.
    pub roots_screened: usize,
    pub trust_rejected: usize,
    pub safety_rejected: usize,
    pub survivors: usize,
    pub probes: usize,
    pub max_q_raw: f64,
    pub max_reliability_raw: f64,
    pub max_satisfied_raw: f64,
}

/// Opaque latent handle for the tensor adapter: the root keeps the encoded
/// state so horizon-1 rolls can use the frame-grounded forward path.
#[derive(Clone)]
pub struct PhaseALatent {
    tensor: Tensor,
    is_root: bool,
}

pub struct TensorPhaseAAdapter<'a> {
    model: &'a WorldModel,
    device: &'a Device,
    physical_batch: usize,
    budget: EvalBudget,
    current_frames: Option<Tensor>,
    /// ADR 0005 §6.1 Channel A window for the decision being scored.
    context: Vec<ContextTransition>,
}

impl<'a> TensorPhaseAAdapter<'a> {
    pub fn new(model: &'a WorldModel, device: &'a Device, physical_batch: usize) -> Self {
        Self {
            model,
            device,
            physical_batch: physical_batch.max(1),
            budget: EvalBudget::default(),
            current_frames: None,
            context: Vec::new(),
        }
    }

    fn backend<T>(result: Result<T>) -> Result<T, ModelCallError> {
        result.map_err(|error| ModelCallError::Backend(format!("{error:#}")))
    }
}

fn parse_action_key(key: &ActionKey) -> Option<ArcAction> {
    let mut parts = key.0.split(':');
    let id = parts.next()?.parse::<u8>().ok()?;
    let x = parts.next().and_then(|v| v.parse::<u8>().ok());
    let y = parts.next().and_then(|v| v.parse::<u8>().ok());
    ArcAction::new(id, x, y).ok()
}

/// Load the Phase A calibration artifact, or fail closed when absent.
pub fn load_phase_a_calibration(path: Option<&std::path::Path>) -> Result<PhaseACalibration> {
    match path {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read Phase A calibration {}", path.display()))?;
            PhaseACalibration::from_json(&text).map_err(|error| {
                anyhow::anyhow!("invalid Phase A calibration {}: {error}", path.display())
            })
        }
        None => Ok(PhaseACalibration::fail_closed()),
    }
}

pub fn phase_a_action_key(action: &ArcAction) -> ActionKey {
    match (action.x, action.y) {
        (Some(x), Some(y)) => ActionKey(format!("{}:{x}:{y}", action.id)),
        _ => ActionKey(format!("{}", action.id)),
    }
}

impl PhaseAModel for TensorPhaseAAdapter<'_> {
    type Latent = PhaseALatent;

    fn encode(&mut self, frame_pixels: &[u8]) -> Result<Self::Latent, ModelCallError> {
        self.budget.charge(ENCODE_EVALS)?;
        let frame = Self::backend(ArcFrame::new(
            FRAME_SIDE as u16,
            FRAME_SIDE as u16,
            frame_pixels.to_vec(),
        ))?;
        let frames = Self::backend(frames_to_indices(std::slice::from_ref(&frame), self.device))?;
        let encoded = Self::backend(self.model.encode_state(&frames))?;
        self.current_frames = Some(frames);
        Ok(PhaseALatent {
            tensor: encoded,
            is_root: true,
        })
    }

    fn step_batch(
        &mut self,
        from: &Self::Latent,
        actions: &[ActionKey],
        goal_vectors: &[[f32; 19]],
    ) -> Result<Vec<StepPrediction<Self::Latent>>, ModelCallError> {
        if actions.is_empty() {
            return Ok(Vec::new());
        }
        self.budget.charge(actions.len())?;
        let parsed = actions
            .iter()
            .map(|key| {
                parse_action_key(key).ok_or_else(|| {
                    ModelCallError::Backend(format!("unparseable action key {}", key.0))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (_, channels, height, width) = Self::backend(from.tensor.dims4().map_err(Into::into))?;
        let mut predictions = Vec::with_capacity(actions.len());
        for (chunk, keys) in parsed
            .chunks(self.physical_batch)
            .zip(actions.chunks(self.physical_batch))
        {
            let n = chunk.len();
            let output = Self::backend((|| -> Result<_> {
                let action_ids = Tensor::from_vec(
                    chunk.iter().map(|a| u32::from(a.id)).collect::<Vec<_>>(),
                    n,
                    self.device,
                )?;
                let coords = Tensor::from_vec(
                    chunk
                        .iter()
                        .flat_map(|a| {
                            [
                                a.x.map_or(0.0, |x| f32::from(x) / 63.0),
                                a.y.map_or(0.0, |y| f32::from(y) / 63.0),
                            ]
                        })
                        .collect::<Vec<_>>(),
                    (n, 2),
                    self.device,
                )?;
                // Goal dropout keeps the transition itself goal-free; goals
                // only enter the event head below (dynamics are goal-independent).
                let zero_goals = Tensor::zeros((n, GOAL_FEATURES_DIM), DType::F32, self.device)?;
                let operator = unknown_operator_conditioning(n, self.device)?;
                let state = from.tensor.broadcast_as((n, channels, height, width))?;
                // Channel A: screening and verification alike see the window.
                let context = context_batch_for(&self.context, n, self.device)?;
                if from.is_root {
                    let frames = self
                        .current_frames
                        .as_ref()
                        .context("root latent has no encoded frame")?;
                    let frame_batch = frames.broadcast_as((n, 1, FRAME_SIDE, FRAME_SIDE))?;
                    self.model
                        .forward_from_encoded_state_with_operator_conditioning_with_context(
                            &state,
                            &frame_batch,
                            &action_ids,
                            &coords,
                            &zero_goals,
                            &operator,
                            context.as_ref(),
                            RecursionDepth::from_config(self.model.config()),
                            0.0,
                            None,
                            RecursionOpts::EVAL,
                        )
                } else {
                    self.model
                        .forward_from_latent_with_depth_and_operator_conditioning_with_context(
                            &state,
                            &action_ids,
                            &coords,
                            &zero_goals,
                            &operator,
                            context.as_ref(),
                            RecursionDepth::from_config(self.model.config()),
                        )
                }
            })())?;
            let q = Self::backend((|| -> Result<Vec<f32>> {
                Ok(ops::sigmoid(&output.q_logit)?
                    .flatten_all()?
                    .to_vec1::<f32>()?)
            })())?;
            let reliability = Self::backend((|| -> Result<Vec<f32>> {
                Ok(ops::sigmoid(&output.reliability_logit)?
                    .flatten_all()?
                    .to_vec1::<f32>()?)
            })())?;
            let noop = Self::backend((|| -> Result<Vec<f32>> {
                Ok(
                    ops::sigmoid(&output.event_logits.narrow(1, EVENT_NOOP, 1)?)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                )
            })())?;
            // Event-head fan-out across goal vectors: reported, not charged.
            let mut per_goal: Vec<Vec<GoalEventReadout>> = vec![Vec::new(); n];
            for goal in goal_vectors {
                let events = Self::backend((|| -> Result<Vec<f32>> {
                    let goal_tensor =
                        Tensor::from_slice(goal, (1, GOAL_FEATURES_DIM), self.device)?
                            .broadcast_as((n, GOAL_FEATURES_DIM))?
                            .contiguous()?;
                    let logits = self.model.event_logits_from(&output.y, &goal_tensor)?;
                    Ok(ops::sigmoid(&logits)?.flatten_all()?.to_vec1::<f32>()?)
                })())?;
                let stride = events.len() / n;
                for (index, slot) in per_goal.iter_mut().enumerate() {
                    let base = index * stride;
                    let read = |event: usize| events.get(base + event).copied().unwrap_or(0.0);
                    slot.push(GoalEventReadout {
                        ordinary: 1.0 - read(EVENT_GOAL_SATISFIED).max(read(EVENT_GOAL_FAILED)),
                        satisfied: read(EVENT_GOAL_SATISFIED),
                        failed: read(EVENT_GOAL_FAILED),
                        exhausted: read(EVENT_EXHAUSTED),
                    });
                }
                self.budget.event_head_reads += n;
            }
            for index in 0..n {
                let latent = Self::backend(output.y.narrow(0, index, 1).map_err(Into::into))?;
                predictions.push(StepPrediction {
                    action: keys[index].clone(),
                    latent: PhaseALatent {
                        tensor: latent,
                        is_root: false,
                    },
                    q_raw: q[index],
                    reliability_raw: reliability[index],
                    noop_raw: noop[index],
                    per_goal_events: std::mem::take(&mut per_goal[index]),
                });
            }
        }
        Ok(predictions)
    }

    fn decode(&mut self, latent: &Self::Latent) -> Result<Vec<u8>, ModelCallError> {
        self.budget.charge(1)?;
        let frames = self
            .current_frames
            .as_ref()
            .ok_or_else(|| ModelCallError::Backend("decode before encode".into()))?;
        Self::backend((|| -> Result<Vec<u8>> {
            let decoded = self
                .model
                .composed_gameplay_decode(&latent.tensor, frames)?;
            Ok(decoded
                .flatten_all()?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|value| value.min(255) as u8)
                .collect())
        })())
    }

    fn evals_used(&self) -> usize {
        self.budget.used
    }

    fn event_head_reads(&self) -> usize {
        self.budget.event_head_reads
    }

    fn reset_decision_budget(&mut self, cap: usize) {
        self.budget.reset(cap);
    }

    fn set_context_window(&mut self, window: Vec<ContextTransition>) {
        self.context = window;
    }

    fn whole_frame(&self) -> bool {
        self.model.config().world_core_v6
    }
}

struct PendingTransition {
    prev_id: RawObservationId,
    prev_pixels: Vec<u8>,
    prev_levels: u16,
    key: ActionKey,
    q_raw: f32,
    reliability_raw: f32,
    per_goal_events: Vec<GoalEventReadout>,
    candidate_keys: Vec<String>,
    decoded: Option<Vec<u8>>,
}

struct Prefix<L> {
    keys: Vec<ActionKey>,
    latent: L,
    q_raw: f32,
    reliability_raw: f32,
    noop_sum: f64,
    per_goal_events: Vec<GoalEventReadout>,
}

pub struct PhaseAPolicy<M: PhaseAModel> {
    adapter: M,
    config: PhaseAConfig,
    calibration: PhaseACalibration,
    action6_max_candidates: usize,
    action6_grid_stride: usize,
    graph: ObservedStateGraph,
    belief: Option<BeliefState>,
    candidates: Option<CandidateSet>,
    pending: Option<PendingTransition>,
    prev_inventory: Option<FeatureInventory>,
    /// ADR 0005 §6.1 Channel A state; disabled unless [`Self::set_context`].
    context: LiveContext,
}

impl<'a> PhaseAPolicy<TensorPhaseAAdapter<'a>> {
    /// Fail-closed constructor: every Phase A inference capability must pass.
    pub fn with_tensor_model(
        model: &'a WorldModel,
        device: &'a Device,
        physical_batch: usize,
        config: PhaseAConfig,
        calibration: PhaseACalibration,
        action6_max_candidates: usize,
        action6_grid_stride: usize,
    ) -> Result<Self> {
        config.validate()?;
        let capabilities = model.phase_a_inference_capabilities();
        let failures = [
            ("patch4_grid", &capabilities.patch4_grid),
            (
                "spatial_prefix_faithful",
                &capabilities.spatial_prefix_faithful,
            ),
            ("action_faithful_ptrm", &capabilities.action_faithful_ptrm),
            (
                "composed_decode_available",
                &capabilities.composed_decode_available,
            ),
            (
                "null_action_row_present",
                &capabilities.null_action_row_present,
            ),
        ]
        .into_iter()
        .filter(|(_, check)| !check.passed)
        .map(|(name, check)| format!("{name}: {}", check.reason.clone().unwrap_or_default()))
        .collect::<Vec<_>>();
        if !failures.is_empty() {
            bail!(
                "Phase A blocked by inference capability checks: {}",
                failures.join("; ")
            );
        }
        Ok(Self::new(
            TensorPhaseAAdapter::new(model, device, physical_batch),
            config,
            calibration,
            action6_max_candidates,
            action6_grid_stride,
        ))
    }
}

impl<M: PhaseAModel> PhaseAPolicy<M> {
    pub fn new(
        adapter: M,
        config: PhaseAConfig,
        calibration: PhaseACalibration,
        action6_max_candidates: usize,
        action6_grid_stride: usize,
    ) -> Self {
        Self {
            adapter,
            config,
            calibration,
            action6_max_candidates,
            action6_grid_stride,
            graph: ObservedStateGraph::default(),
            belief: None,
            candidates: None,
            pending: None,
            prev_inventory: None,
            context: LiveContext::disabled(),
        }
    }

    /// Channel A state (ADR 0005 §6.1); see `arc3_live::live_context_for`.
    pub fn set_context(&mut self, context: LiveContext) {
        self.context = context;
    }

    fn clear_episode_state(&mut self) {
        self.graph = ObservedStateGraph::default();
        self.belief = None;
        self.candidates = None;
        self.pending = None;
        self.prev_inventory = None;
    }

    fn raw_id(observation: &ArcObservation, pixels: &[u8]) -> RawObservationId {
        let mut hasher = Sha256::new();
        hasher.update(observation.game_id.as_bytes());
        hasher.update(observation.levels_completed.to_le_bytes());
        hasher.update(pixels);
        RawObservationId::new(hasher.finalize().into())
    }

    fn observed_channel(observation: &ArcObservation, prev_levels: u16) -> TerminalChannel {
        if observation.levels_completed > prev_levels {
            TerminalChannel::Satisfied
        } else if observation.state == "GAME_OVER" {
            TerminalChannel::Failed
        } else {
            TerminalChannel::Ordinary
        }
    }

    fn gameplay_matches(decoded: &[u8], observed: &[u8]) -> bool {
        // Composed decode covers rows 0..63; the status row is excluded.
        let rows = decoded.len().min(63 * FRAME_SIDE);
        decoded[..rows] == observed[..rows]
    }

    /// Fold the previous real transition into the graph and posterior.
    fn observe_pending(
        &mut self,
        observation: &ArcObservation,
        id: RawObservationId,
        pixels: &[u8],
    ) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let channel = Self::observed_channel(observation, pending.prev_levels);
        let noop = pending.prev_pixels == pixels;
        let board_effect = pending
            .prev_pixels
            .iter()
            .zip(pixels)
            .map(|(before, after)| u8::from(before != after))
            .collect::<Vec<_>>();
        let exact_match = pending
            .decoded
            .as_deref()
            .is_some_and(|decoded| Self::gameplay_matches(decoded, pixels));
        let edge = FactualEdge {
            action: pending.key.clone(),
            next_raw_id: Some(id),
            board_effect,
            terminal: channel.clone(),
            action_cost: 1,
            model_prediction: ModelPredictionRecord {
                decoded_gameplay_frame: pending.decoded.clone(),
                trusted: true,
            },
        };
        match self.graph.append_edge(pending.prev_id, edge) {
            Ok(()) => {
                let _ = self
                    .graph
                    .retrodict(pending.prev_id, &pending.key, exact_match);
            }
            Err(ObservedGraphError::EdgeAlreadyRecorded(_)) => {}
            Err(_) => {}
        }
        let (Some(belief), Some(candidates)) = (self.belief.as_mut(), self.candidates.as_ref())
        else {
            return;
        };
        let keys = candidate_keys(candidates);
        if keys != pending.candidate_keys || pending.per_goal_events.len() < keys.len() {
            return;
        }
        let eta =
            match self
                .calibration
                .edge_trust(pending.q_raw, pending.reliability_raw, &self.config)
            {
                EdgeTrust::Trusted { eta } => eta,
                EdgeTrust::Untrusted => 0.0,
            };
        let likelihoods = pending
            .per_goal_events
            .iter()
            .take(keys.len())
            .map(|events| {
                let base = match channel {
                    TerminalChannel::Satisfied => events.satisfied,
                    TerminalChannel::Failed => events.failed,
                    TerminalChannel::Exhausted => events.exhausted,
                    TerminalChannel::Ordinary => events.ordinary,
                };
                let base = f64::from(base);
                if noop {
                    base
                } else {
                    base.max(f64::EPSILON)
                }
            })
            .collect::<Vec<_>>();
        let _ = belief.soft_update(&likelihoods, eta, self.calibration.tau_unknown);
    }

    fn refresh_candidates(&mut self, inventory: &FeatureInventory) {
        let background = inventory
            .palette_counts
            .iter()
            .max_by_key(|(color, count)| (**count, std::cmp::Reverse(**color)))
            .map(|(color, _)| *color)
            .unwrap_or(0);
        let mut proposals = propose_candidates(inventory, self.prev_inventory.as_ref(), background);
        if let Some(existing) = &self.candidates {
            proposals.extend(existing.restore_dormant());
        }
        let unknown_prior = self.config.unknown_prior;
        let candidates =
            match build_candidate_set(proposals, self.config.max_candidates, unknown_prior) {
                Ok(set) if !set.candidates.is_empty() => set,
                _ => {
                    self.candidates = None;
                    self.belief = None;
                    return;
                }
            };
        let new_keys = candidate_keys(&candidates);
        let reuse = self
            .candidates
            .as_ref()
            .is_some_and(|old| candidate_keys(old) == new_keys);
        if !reuse || self.belief.is_none() {
            // `concrete_masses` are f32; their f64 widening misses the belief
            // validator's 1e-9 sum tolerance, so renormalize in f64 against the
            // unknown mass before constructing the posterior.
            let unknown = f64::from(candidates.unknown_mass);
            let raw = candidates
                .concrete_masses
                .iter()
                .map(|mass| f64::from(*mass))
                .collect::<Vec<_>>();
            let total: f64 = raw.iter().sum();
            self.belief = if total > 0.0 && unknown > 0.0 && unknown < 1.0 {
                let scale = (1.0 - unknown) / total;
                let masses = raw.into_iter().map(|mass| mass * scale).collect::<Vec<_>>();
                BeliefState::new(masses, unknown).ok()
            } else {
                None
            };
        }
        self.candidates = Some(candidates);
    }

    fn frontier_action(
        &self,
        id: RawObservationId,
        legal: &[ActionKey],
    ) -> (Option<ActionKey>, Vec<String>) {
        if let Some(frontier) = self.graph.nearest_untried_frontier(id) {
            if frontier.prefix.is_empty() {
                return (frontier.untried_actions.first().cloned(), Vec::new());
            }
            // A frontier behind a factual prefix: replay its first step.
            return (
                frontier.prefix.first().cloned(),
                frontier.prefix.iter().map(|k| k.0.clone()).collect(),
            );
        }
        (legal.first().cloned(), Vec::new())
    }

    fn passes_safety(
        &self,
        events: &[GoalEventReadout],
        protected: &[usize],
        protect_unknown: bool,
        goal_count: usize,
    ) -> bool {
        let alpha = f64::from(self.config.alpha_safe);
        let mut indices = protected.to_vec();
        if protect_unknown {
            indices.push(goal_count); // the zero-goal safety particle
        }
        indices.into_iter().all(|index| {
            events
                .get(index)
                .is_none_or(|e| f64::from(e.failed) + f64::from(e.exhausted) <= alpha)
        })
    }

    fn claim_potential(events: &[GoalEventReadout], goal_count: usize) -> (usize, f64) {
        let claimants = events
            .iter()
            .take(goal_count)
            .filter(|e| e.satisfied >= CLAIM_RAW_THRESHOLD)
            .count();
        let mass: f64 = events
            .iter()
            .take(goal_count)
            .map(|e| f64::from(e.satisfied))
            .sum();
        (claimants, mass)
    }
}

fn candidate_keys(set: &CandidateSet) -> Vec<String> {
    set.candidates
        .iter()
        .map(|c| format!("{}|{}", c.family, c.predicate_id))
        .collect()
}

fn plain_frame(pixels: &[u8]) -> Option<PlainFrame> {
    let mut palette = pixels.to_vec();
    palette.sort_unstable();
    palette.dedup();
    PlainFrame::new(pixels.to_vec(), palette).ok()
}

impl<M: PhaseAModel> LivePolicy for PhaseAPolicy<M> {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
        let started = Instant::now();
        self.adapter
            .reset_decision_budget(self.config.max_model_evals);
        // Channel A: only transitions confirmed before this decision.
        let window = self.context.window(observation.levels_completed);
        let context_len = window.len();
        self.adapter.set_context_window(window);
        let pixels: Vec<u8> = observation.frame.pixels.to_vec();
        if pixels.len() != FRAME_PIXELS {
            bail!(
                "Phase A expects a {FRAME_SIDE}x{FRAME_SIDE} canvas, got {} pixels",
                pixels.len()
            );
        }
        let id = Self::raw_id(observation, &pixels);
        // ADR 0005 §1.3: v6 frames are whole-frame content, so ACTION6 may
        // target row 63 (mirrors the greedy `ModelPolicy`).
        let actions = enumerate_actions_with(
            observation,
            self.action6_max_candidates,
            self.action6_grid_stride,
            self.adapter.whole_frame(),
        )?;
        if actions.is_empty() {
            bail!("Phase A has no legal actions to choose from");
        }
        let key_to_action: BTreeMap<ActionKey, ArcAction> = actions
            .iter()
            .map(|action| (phase_a_action_key(action), action.clone()))
            .collect();
        let legal: Vec<ActionKey> = actions.iter().map(phase_a_action_key).collect();
        match self
            .graph
            .insert_node(id, pixels.clone(), legal.iter().cloned())
        {
            Ok(()) | Err(ObservedGraphError::ObservationCollision(_)) => {}
            Err(error) => bail!("observed-state graph rejected node: {error}"),
        }
        self.observe_pending(observation, id, &pixels);

        let inventory = plain_frame(&pixels).map(|frame| feature_inventory(&frame));
        if let Some(inventory) = &inventory {
            self.refresh_candidates(inventory);
        }
        let goal_count = self.candidates.as_ref().map_or(0, |c| c.candidates.len());
        let unknown_mass = self.belief.as_ref().map_or(1.0, |b| b.unknown_mass);
        let protected = self
            .belief
            .as_ref()
            .map(BeliefState::protected_indices)
            .unwrap_or_default();
        let calibration_missing = self.calibration.uncalibrated;
        let target_ms = self.config.deadline.target_millis;

        let mut mode = "fail_closed_frontier";
        let mut chosen_key: Option<ActionKey> = None;
        let mut chosen_prefix: Vec<String> = Vec::new();
        let mut claim_mass = 0.0;
        let mut deadline_truncated = false;
        let mut chosen_q = 0.0f32;
        let mut chosen_rel = 0.0f32;
        let mut chosen_noop = 0.0f32;
        let mut chosen_events: Vec<GoalEventReadout> = Vec::new();
        let mut chosen_decoded: Option<Vec<u8>> = None;
        let mut roots_screened = 0usize;
        let mut trust_rejected = 0usize;
        let mut safety_rejected = 0usize;
        let mut probe_count = 0usize;
        let mut max_q_raw = 0.0f64;
        let mut max_rel_raw = 0.0f64;
        let mut max_sat_raw = 0.0f64;

        let trusted_path = !calibration_missing && goal_count > 0 && self.belief.is_some();
        if trusted_path {
            let goal_vectors: Vec<[f32; 19]> = {
                let candidates = self.candidates.as_ref().expect("candidates present");
                let mut vectors = candidates
                    .candidates
                    .iter()
                    .map(|c| c.g19)
                    .collect::<Vec<_>>();
                vectors.push([0.0; 19]);
                vectors
            };
            let protect_unknown = unknown_mass > UNKNOWN_PROTECT_THRESHOLD;
            match self.adapter.encode(&pixels) {
                Ok(root) => {
                    // Horizon-1 screening: simple actions first, then ACTION6 sites.
                    let mut roots: Vec<ActionKey> = legal
                        .iter()
                        .filter(|k| !k.0.contains(':'))
                        .cloned()
                        .collect();
                    roots.extend(legal.iter().filter(|k| k.0.contains(':')).cloned());
                    roots.truncate(ROOT_SCREEN_EVALS);
                    let mut survivors: Vec<Prefix<M::Latent>> = Vec::new();
                    if let Ok(predictions) = self.adapter.step_batch(&root, &roots, &goal_vectors) {
                        for prediction in predictions {
                            roots_screened += 1;
                            max_q_raw = max_q_raw.max(f64::from(prediction.q_raw));
                            max_rel_raw = max_rel_raw.max(f64::from(prediction.reliability_raw));
                            max_sat_raw = max_sat_raw.max(
                                prediction
                                    .per_goal_events
                                    .iter()
                                    .take(goal_count)
                                    .map(|e| f64::from(e.satisfied))
                                    .fold(0.0, f64::max),
                            );
                            let trusted = matches!(
                                self.calibration.edge_trust(
                                    prediction.q_raw,
                                    prediction.reliability_raw,
                                    &self.config
                                ),
                                EdgeTrust::Trusted { .. }
                            );
                            if !trusted {
                                trust_rejected += 1;
                                continue;
                            }
                            if !self.passes_safety(
                                &prediction.per_goal_events,
                                &protected,
                                protect_unknown,
                                goal_count,
                            ) {
                                safety_rejected += 1;
                                continue;
                            }
                            survivors.push(Prefix {
                                keys: vec![prediction.action.clone()],
                                latent: prediction.latent.clone(),
                                q_raw: prediction.q_raw,
                                reliability_raw: prediction.reliability_raw,
                                noop_sum: f64::from(prediction.noop_raw),
                                per_goal_events: prediction.per_goal_events,
                            });
                        }
                    }
                    survivors.sort_by(|a, b| {
                        let (ca, ma) = Self::claim_potential(&a.per_goal_events, goal_count);
                        let (cb, mb) = Self::claim_potential(&b.per_goal_events, goal_count);
                        cb.cmp(&ca)
                            .then_with(|| mb.total_cmp(&ma))
                            .then_with(|| a.keys.cmp(&b.keys))
                    });
                    // Horizon-2 verification on the top finalists, receding.
                    let simple: Vec<ActionKey> = legal
                        .iter()
                        .filter(|k| !k.0.contains(':'))
                        .take(MAX_EXTENSIONS_PER_FINALIST)
                        .cloned()
                        .collect();
                    let mut verified: Vec<Prefix<M::Latent>> = Vec::new();
                    let mut verify_used = 0usize;
                    if self.config.max_horizon >= 2 {
                        for finalist in survivors.iter().take(MAX_FINALISTS) {
                            if started.elapsed().as_millis() as u64 > target_ms {
                                deadline_truncated = true;
                                break;
                            }
                            let room = VERIFY_EVALS.saturating_sub(verify_used);
                            let extensions: Vec<ActionKey> =
                                simple.iter().take(room).cloned().collect();
                            if extensions.is_empty() {
                                break;
                            }
                            let Ok(predictions) = self.adapter.step_batch(
                                &finalist.latent,
                                &extensions,
                                &goal_vectors,
                            ) else {
                                break;
                            };
                            verify_used += extensions.len();
                            for prediction in predictions {
                                let trusted = matches!(
                                    self.calibration.edge_trust(
                                        prediction.q_raw,
                                        prediction.reliability_raw,
                                        &self.config
                                    ),
                                    EdgeTrust::Trusted { .. }
                                );
                                if !trusted
                                    || !self.passes_safety(
                                        &prediction.per_goal_events,
                                        &protected,
                                        protect_unknown,
                                        goal_count,
                                    )
                                {
                                    continue;
                                }
                                let mut keys = finalist.keys.clone();
                                keys.push(prediction.action.clone());
                                verified.push(Prefix {
                                    keys,
                                    latent: prediction.latent.clone(),
                                    q_raw: finalist.q_raw,
                                    reliability_raw: finalist.reliability_raw,
                                    noop_sum: finalist.noop_sum + f64::from(prediction.noop_raw),
                                    per_goal_events: prediction.per_goal_events,
                                });
                            }
                        }
                    }
                    let mut prefixes = survivors;
                    prefixes.extend(verified);
                    prefixes.sort_by(|a, b| {
                        let (ca, ma) = Self::claim_potential(&a.per_goal_events, goal_count);
                        let (cb, mb) = Self::claim_potential(&b.per_goal_events, goal_count);
                        cb.cmp(&ca)
                            .then_with(|| mb.total_cmp(&ma))
                            .then_with(|| a.keys.len().cmp(&b.keys.len()))
                            .then_with(|| a.keys.cmp(&b.keys))
                    });
                    // Claims require the executable predicate on the decoded
                    // endpoint plus a calibrated satisfaction lower bound.
                    let belief = self.belief.as_ref().expect("belief present");
                    let candidates = self.candidates.as_ref().expect("candidates present");
                    let min_claim = f64::from(self.config.epsilon);
                    let mut probes = Vec::new();
                    let mut decoded_by_prefix: BTreeMap<Vec<ActionKey>, Vec<u8>> = BTreeMap::new();
                    let mut decodes = 0usize;
                    for prefix in prefixes.iter().take(MAX_DECODED_PREFIXES) {
                        if decodes >= DECODE_EVALS {
                            break;
                        }
                        let Ok(decoded) = self.adapter.decode(&prefix.latent) else {
                            break;
                        };
                        decodes += 1;
                        let mut end_pixels = decoded.clone();
                        end_pixels.resize(FRAME_PIXELS, 0);
                        end_pixels[63 * FRAME_SIDE..].copy_from_slice(&pixels[63 * FRAME_SIDE..]);
                        let Some(end_inventory) =
                            plain_frame(&end_pixels).map(|f| feature_inventory(&f))
                        else {
                            continue;
                        };
                        let start_inventory = inventory.as_ref().expect("inventory present");
                        let claims = candidates
                            .candidates
                            .iter()
                            .enumerate()
                            .filter_map(|(index, candidate)| {
                                let events = prefix.per_goal_events.get(index)?;
                                let lcb = self.calibration.satisfaction_lcb(events.satisfied)?;
                                if events.satisfied < CLAIM_RAW_THRESHOLD
                                    || !evaluate_predicate(
                                        candidate,
                                        start_inventory,
                                        &end_inventory,
                                    )
                                {
                                    return None;
                                }
                                Some(ProbeClaim {
                                    candidate_index: index,
                                    posterior_mass: belief
                                        .concrete_weights
                                        .get(index)
                                        .copied()
                                        .unwrap_or(0.0),
                                    satisfaction_lcb: lcb,
                                    protected: protected.contains(&index),
                                })
                            })
                            .collect::<Vec<_>>();
                        decoded_by_prefix.insert(prefix.keys.clone(), decoded);
                        let probe = CandidateProbe {
                            actions: prefix.keys.clone(),
                            safe: true,
                            claims,
                            summed_noop_probability: prefix.noop_sum,
                            graph_repeats: 0,
                        };
                        if accept_finalist(
                            probe.claim_mass(),
                            min_claim,
                            self.calibration.score_error_bound,
                        ) {
                            probes.push(probe);
                        }
                        probe_count += 1;
                    }
                    let frontier = self.graph.nearest_untried_frontier(id);
                    match choose_probe(probes, frontier) {
                        Some(ProbeChoice::MultiGoal(probe))
                        | Some(ProbeChoice::SingleGoal(probe)) => {
                            mode = "goal_probe";
                            claim_mass = probe.claim_mass();
                            chosen_prefix = probe.actions.iter().map(|k| k.0.clone()).collect();
                            chosen_key = probe.actions.first().cloned();
                            if let Some(prefix) = prefixes.iter().find(|p| p.keys == probe.actions)
                            {
                                chosen_q = prefix.q_raw;
                                chosen_rel = prefix.reliability_raw;
                                chosen_noop = prefix.noop_sum as f32;
                                chosen_events = prefix.per_goal_events.clone();
                            }
                            if probe.actions.len() == 1 {
                                chosen_decoded = decoded_by_prefix.remove(&probe.actions);
                            }
                        }
                        Some(ProbeChoice::GraphFrontier(frontier)) => {
                            mode = "graph_frontier";
                            let (key, prefix) = if frontier.prefix.is_empty() {
                                (frontier.untried_actions.first().cloned(), Vec::new())
                            } else {
                                (
                                    frontier.prefix.first().cloned(),
                                    frontier.prefix.iter().map(|k| k.0.clone()).collect(),
                                )
                            };
                            chosen_key = key;
                            chosen_prefix = prefix;
                        }
                        None => {
                            mode = "no_probe";
                        }
                    }
                    if let Some(key) = &chosen_key {
                        if let Some(prefix) = prefixes
                            .iter()
                            .find(|p| p.keys.len() == 1 && &p.keys[0] == key)
                        {
                            chosen_q = prefix.q_raw;
                            chosen_rel = prefix.reliability_raw;
                            chosen_noop = prefix.noop_sum as f32;
                            chosen_events = prefix.per_goal_events.clone();
                        }
                    }
                }
                Err(ModelCallError::BudgetExhausted { .. }) => {
                    mode = "budget_refused";
                }
                Err(error) => bail!("Phase A encode failed: {error}"),
            }
        }
        if chosen_key.is_none() {
            let (key, prefix) = self.frontier_action(id, &legal);
            chosen_key = key;
            chosen_prefix = prefix;
            if mode == "goal_probe" {
                mode = "graph_frontier";
            }
        }
        let key = chosen_key.context("Phase A produced no action")?;
        let action = key_to_action
            .get(&key)
            .cloned()
            .with_context(|| format!("Phase A chose an action outside the legal set: {}", key.0))?;
        let selection = selection_charge(self.calibration.score_error_bound);
        let trace = PhaseADecisionTrace {
            mode: mode.into(),
            model_evals: self.adapter.evals_used(),
            event_head_reads: self.adapter.event_head_reads(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            goal_candidates: goal_count,
            unknown_mass,
            protected: protected.len(),
            claim_mass,
            selection_charge: selection,
            deadline_truncated,
            calibration_missing,
            prefix: chosen_prefix,
            roots_screened,
            trust_rejected,
            safety_rejected,
            survivors: roots_screened.saturating_sub(trust_rejected + safety_rejected),
            probes: probe_count,
            max_q_raw,
            max_reliability_raw: max_rel_raw,
            max_satisfied_raw: max_sat_raw,
        };
        let candidate_keys_now = self
            .candidates
            .as_ref()
            .map(candidate_keys)
            .unwrap_or_default();
        self.pending = Some(PendingTransition {
            prev_id: id,
            prev_pixels: pixels,
            prev_levels: observation.levels_completed,
            key: key.clone(),
            q_raw: chosen_q,
            reliability_raw: chosen_rel,
            per_goal_events: chosen_events,
            candidate_keys: candidate_keys_now,
            decoded: chosen_decoded,
        });
        self.prev_inventory = inventory;
        Ok(ActionDecision {
            chosen: ActionScore {
                action,
                score: claim_mass,
                q_probability: f64::from(chosen_q),
                reliability_probability: f64::from(chosen_rel),
                noop_probability: f64::from(chosen_noop),
                predicted_effect: 0.0,
            },
            candidate_count: actions.len(),
            phase_a: Some(trace),
            adaptation: None,
            context_len,
        })
    }

    fn policy_name(&self) -> &'static str {
        PHASE_A_POLICY
    }

    fn on_game_start(&mut self, _game_id: &str) {
        self.clear_episode_state();
        self.context.begin_game();
    }

    fn on_confirmed_transition(
        &mut self,
        current: &ArcObservation,
        action: &ArcAction,
        next: &ArcObservation,
    ) {
        self.context.observe(
            &current.frame,
            action,
            &next.frame,
            current.levels_completed,
        );
    }

    fn on_level_transition(&mut self, _levels_completed: u16) {
        self.clear_episode_state();
    }

    fn on_reset_retry(&mut self, _reason: &str) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::latent_planning::adapter::EvalBudget;
    use crate::p2::latent_planning::trust::CalibrationBin;

    /// Deterministic pure-Rust adapter with scripted readouts and a budget.
    struct FakeModel {
        budget: EvalBudget,
        step_calls: usize,
        encode_calls: usize,
        satisfied_for_first_action: f32,
        context_lens: Vec<usize>,
        whole_frame: bool,
    }

    impl FakeModel {
        fn new(satisfied: f32) -> Self {
            Self {
                budget: EvalBudget::default(),
                step_calls: 0,
                encode_calls: 0,
                satisfied_for_first_action: satisfied,
                context_lens: Vec::new(),
                whole_frame: false,
            }
        }
    }

    impl PhaseAModel for FakeModel {
        type Latent = u32;

        fn encode(&mut self, _frame: &[u8]) -> Result<u32, ModelCallError> {
            self.budget.charge(1)?;
            self.encode_calls += 1;
            Ok(0)
        }

        fn step_batch(
            &mut self,
            from: &u32,
            actions: &[ActionKey],
            goal_vectors: &[[f32; 19]],
        ) -> Result<Vec<StepPrediction<u32>>, ModelCallError> {
            self.budget.charge(actions.len())?;
            self.step_calls += 1;
            Ok(actions
                .iter()
                .enumerate()
                .map(|(index, action)| StepPrediction {
                    action: action.clone(),
                    latent: from + 1 + index as u32,
                    q_raw: 0.9,
                    reliability_raw: 0.9,
                    noop_raw: 0.1,
                    per_goal_events: goal_vectors
                        .iter()
                        .map(|_| GoalEventReadout {
                            ordinary: 0.7,
                            satisfied: if index == 0 {
                                self.satisfied_for_first_action
                            } else {
                                0.1
                            },
                            failed: 0.0,
                            exhausted: 0.0,
                        })
                        .collect(),
                })
                .collect())
        }

        fn decode(&mut self, _latent: &u32) -> Result<Vec<u8>, ModelCallError> {
            self.budget.charge(1)?;
            Ok(vec![0u8; 63 * FRAME_SIDE])
        }

        fn evals_used(&self) -> usize {
            self.budget.used
        }

        fn event_head_reads(&self) -> usize {
            self.budget.event_head_reads
        }

        fn reset_decision_budget(&mut self, cap: usize) {
            self.budget.reset(cap);
        }

        fn set_context_window(&mut self, window: Vec<ContextTransition>) {
            self.context_lens.push(window.len());
        }

        fn whole_frame(&self) -> bool {
            self.whole_frame
        }
    }

    fn observation(levels: u16, marker: bool) -> ArcObservation {
        let mut pixels = vec![0u8; FRAME_PIXELS];
        // A big hazard-like region and a tiny marker.
        for y in 10..30 {
            for x in 10..30 {
                pixels[y * FRAME_SIDE + x] = 3;
            }
        }
        if marker {
            pixels[40 * FRAME_SIDE + 40] = 5;
            pixels[40 * FRAME_SIDE + 41] = 5;
        }
        ArcObservation {
            game_id: "test".into(),
            guid: "g".into(),
            frame: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels).unwrap(),
            animation: vec![],
            full_reset: false,
            state: "NOT_FINISHED".into(),
            levels_completed: levels,
            win_levels: 3,
            available_actions: vec![1, 2, 3],
        }
    }

    fn permissive_calibration() -> PhaseACalibration {
        let bin = || {
            Some(CalibrationBin {
                upper_error_bound_95: 0.01,
                support: 1_000,
            })
        };
        PhaseACalibration {
            q_direction: 1,
            tau_unknown: 0.5,
            score_error_bound: 0.001,
            ordinary: bin(),
            event_false_safe: bin(),
            satisfaction: bin(),
            ptrm: bin(),
            uncalibrated: false,
        }
    }

    /// ADR 0005 §1.3: a v6 Phase A policy enumerates ACTION6 coordinates on
    /// row 63; a legacy (v5) policy keeps the reserved status row excluded.
    #[test]
    fn phase_a_action6_proposals_reach_row_63_only_under_v6() -> Result<()> {
        let legal_keys = |whole_frame: bool| -> Result<Vec<String>> {
            let mut model = FakeModel::new(0.9);
            model.whole_frame = whole_frame;
            let mut policy = PhaseAPolicy::new(
                model,
                PhaseAConfig::default(),
                permissive_calibration(),
                64,
                32,
            );
            policy.on_game_start("game");
            let mut observation = observation(0, true);
            observation.available_actions = vec![1, 2, 3, 6];
            // Background colour 5 with a lone palette-0 pixel on row 63 (as in
            // the arc3_live row-63 test): legacy enumeration treats row 63 as
            // the status row and index 0 as padding, so it never proposes the
            // pixel; whole-frame enumeration proposes its component points.
            let mut pixels = observation
                .frame
                .pixels
                .iter()
                .map(|&p| if p == 0 { 5 } else { p })
                .collect::<Vec<u8>>();
            pixels[63 * FRAME_SIDE + 7] = 0;
            observation.frame =
                ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels.clone()).unwrap();
            policy.choose_action(&observation)?;
            let id = PhaseAPolicy::<FakeModel>::raw_id(&observation, &pixels);
            let node = policy.graph.node(id).expect("observed node is recorded");
            Ok(node.legal_actions.iter().map(|k| k.0.clone()).collect())
        };
        let on_row_63 = |keys: &[String]| {
            keys.iter()
                .any(|key| key.starts_with("6:") && key.ends_with(":63"))
        };
        let v6 = legal_keys(true)?;
        assert!(v6.iter().any(|key| key.starts_with("6:")));
        assert!(
            on_row_63(&v6),
            "v6 Phase A must propose ACTION6 on row 63: {v6:?}"
        );
        let v5 = legal_keys(false)?;
        assert!(v5.iter().any(|key| key.starts_with("6:")));
        assert!(
            !on_row_63(&v5),
            "v5 Phase A must not propose row 63: {v5:?}"
        );
        Ok(())
    }

    #[test]
    fn phase_a_policy_hands_the_preceding_window_to_the_adapter() -> Result<()> {
        use crate::p2::adaptation::{AdaptationMode, LiveContext};
        let mut policy = PhaseAPolicy::new(
            FakeModel::new(0.9),
            PhaseAConfig::default(),
            permissive_calibration(),
            8,
            8,
        );
        policy.set_context(LiveContext::new(true, AdaptationMode::Reset));
        policy.on_game_start("game");
        let first = policy.choose_action(&observation(0, true))?;
        assert_eq!(first.context_len, 0);
        assert_eq!(policy.adapter.context_lens, vec![0]);
        for id in 1..=3u8 {
            let current = observation(0, true);
            let next = observation(0, false);
            let action = ArcAction::new(id, None, None)?;
            policy.on_confirmed_transition(&current, &action, &next);
        }
        let decision = policy.choose_action(&observation(0, true))?;
        assert_eq!(decision.context_len, 3);
        assert_eq!(
            policy.adapter.context_lens.last(),
            Some(&3),
            "the window is set before any model call of the decision"
        );
        // Level 1 (default arm) starts without level 0's transitions.
        policy.on_level_transition(1);
        assert_eq!(policy.choose_action(&observation(1, true))?.context_len, 0);
        // Game start empties Factual Memory.
        policy.on_game_start("next-game");
        assert_eq!(policy.choose_action(&observation(0, true))?.context_len, 0);
        Ok(())
    }

    #[test]
    fn tensor_adapter_conditions_screening_and_verification_on_context() -> Result<()> {
        use crate::p2::experiment::ConsumerReadoutTopology;
        use crate::p2::model::ModelConfig;
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let config = ModelConfig {
            patch_size: 4,
            hidden_dim: 8,
            action_dim: 8,
            inner_steps: 1,
            outer_steps: 1,
            spatial_action_field: true,
            world_core_v4: true,
            world_core_v5: true,
            world_core_v6: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(
            config,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        {
            // Zero-initialised context FiLM would hide the window; perturb it.
            let data = varmap.data().lock().unwrap();
            let var = data.get("context_film_gamma.weight").expect("v6 parameter");
            var.set(&var.as_tensor().ones_like()?.affine(0.5, 0.0)?)?;
        }
        let frame = |fill: u8| ArcFrame::new(64, 64, vec![fill; FRAME_PIXELS]).unwrap();
        let window: Vec<ContextTransition> = (0..3u8)
            .map(|i| ContextTransition {
                current: frame(i),
                action: ArcAction::new(1 + i, None, None).unwrap(),
                next: frame(i + 1),
            })
            .collect();
        let keys = vec![
            phase_a_action_key(&ArcAction::new(1, None, None)?),
            phase_a_action_key(&ArcAction::new(6, Some(3), Some(4))?),
        ];
        let goals = [[0.0f32; 19]];

        let mut adapter = TensorPhaseAAdapter::new(&model, &device, 8);
        adapter.reset_decision_budget(64);
        let root = adapter.encode(&frame(7).pixels)?;
        // Screening (root, frame-grounded path).
        let plain = adapter.step_batch(&root, &keys, &goals)?;
        adapter.set_context_window(window.clone());
        let contextual = adapter.step_batch(&root, &keys, &goals)?;
        assert!(
            plain
                .iter()
                .zip(&contextual)
                .any(|(a, b)| a.q_raw != b.q_raw || a.reliability_raw != b.reliability_raw),
            "screening must be conditioned on the window"
        );
        // Verification (non-root latent path) with the same window.
        let from = &contextual[0].latent;
        let verified = adapter.step_batch(from, &keys, &goals)?;
        adapter.set_context_window(Vec::new());
        let verified_plain = adapter.step_batch(from, &keys, &goals)?;
        assert!(
            verified
                .iter()
                .zip(&verified_plain)
                .any(|(a, b)| a.q_raw != b.q_raw || a.reliability_raw != b.reliability_raw),
            "verification must be conditioned on the window"
        );
        for prediction in verified.iter().chain(&contextual) {
            assert!(prediction.q_raw.is_finite() && prediction.reliability_raw.is_finite());
        }
        Ok(())
    }

    #[test]
    fn fail_closed_calibration_never_touches_the_model() -> Result<()> {
        let mut policy = PhaseAPolicy::new(
            FakeModel::new(0.9),
            PhaseAConfig::default(),
            PhaseACalibration::fail_closed(),
            8,
            8,
        );
        let decision = policy.choose_action(&observation(0, true))?;
        let trace = decision.phase_a.expect("phase-a trace");
        assert_eq!(trace.mode, "fail_closed_frontier");
        assert!(trace.calibration_missing);
        assert_eq!(policy.adapter.encode_calls, 0);
        assert_eq!(policy.adapter.step_calls, 0);
        assert_eq!(trace.model_evals, 0);
        Ok(())
    }

    #[test]
    fn budget_and_horizon_invariants_hold_across_a_game() -> Result<()> {
        let mut policy = PhaseAPolicy::new(
            FakeModel::new(0.9),
            PhaseAConfig::default(),
            permissive_calibration(),
            8,
            8,
        );
        for step in 0..12u16 {
            let decision = policy.choose_action(&observation(step / 4, true))?;
            let trace = decision.phase_a.expect("trace");
            assert!(trace.model_evals <= PhaseAConfig::default().max_model_evals);
            assert!(
                trace.prefix.len() <= 2,
                "horizon exceeded: {:?}",
                trace.prefix
            );
            assert_ne!(trace.mode, "fail_closed_frontier");
        }
        Ok(())
    }

    #[test]
    fn graph_frontier_fallback_when_nothing_claims() -> Result<()> {
        let mut policy = PhaseAPolicy::new(
            FakeModel::new(0.0),
            PhaseAConfig::default(),
            permissive_calibration(),
            8,
            8,
        );
        let first = policy.choose_action(&observation(0, true))?;
        let trace = first.phase_a.expect("trace");
        assert!(matches!(trace.mode.as_str(), "graph_frontier" | "no_probe"));
        // A second decision on the same observation must not repeat the
        // recorded action: the frontier advances.
        let second = policy.choose_action(&observation(0, true))?;
        assert_ne!(first.chosen.action, second.chosen.action);
        Ok(())
    }

    #[test]
    fn level_transition_clears_episode_state() -> Result<()> {
        let mut policy = PhaseAPolicy::new(
            FakeModel::new(0.9),
            PhaseAConfig::default(),
            permissive_calibration(),
            8,
            8,
        );
        policy.choose_action(&observation(0, true))?;
        assert!(policy.pending.is_some());
        policy.on_level_transition(1);
        assert!(policy.pending.is_none() && policy.belief.is_none());
        Ok(())
    }
}
