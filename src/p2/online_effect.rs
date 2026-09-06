//! Small online predictor for action-conditioned observed changes.
//!
//! The learner is deliberately game-agnostic. It consumes only confirmed
//! `(current frame, action, next frame)` tuples, keeps a bounded in-memory
//! replay, and is freshly seeded by its caller. A high change probability is
//! an exploration signal, not a reward or value estimate.
//!
//! ACTION6's coordinate is a single spatial impulse. The three `3x3`
//! convolutions give it a local `7x7` receptive field, so this prerequisite
//! model cannot condition remote visible effects directly on the click site.

use crate::p2::data::{ArcAction, ArcFrame, FRAME_SIDE};
use anyhow::{bail, ensure, Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, ParamsAdamW};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, Optimizer, VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

const PALETTE_CHANNELS: usize = 16;
const ACTION_CHANNELS: usize = 7;
const COORDINATE_CHANNELS: usize = 1;
const INPUT_CHANNELS: usize = PALETTE_CHANNELS + ACTION_CHANNELS + COORDINATE_CHANNELS;
const FRAME_PIXELS: usize = FRAME_SIDE * FRAME_SIDE;

/// Resource and optimizer bounds for one fresh online learner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnlineEffectConfig {
    /// Width of each hidden convolution.
    pub hidden_channels: usize,
    /// Maximum number of factual transitions retained in memory.
    pub replay_capacity: usize,
    /// Number of most-recent replay rows used by one optimizer update.
    pub physical_batch: usize,
    pub learning_rate: f64,
}

impl Default for OnlineEffectConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 16,
            replay_capacity: 256,
            physical_batch: 16,
            learning_rate: 1e-3,
        }
    }
}

impl OnlineEffectConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.hidden_channels > 0, "hidden_channels must be positive");
        ensure!(self.replay_capacity > 0, "replay_capacity must be positive");
        ensure!(self.physical_batch > 0, "physical_batch must be positive");
        ensure!(
            self.physical_batch <= self.replay_capacity,
            "physical_batch {} exceeds replay_capacity {}",
            self.physical_batch,
            self.replay_capacity
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "learning_rate must be positive and finite"
        );
        Ok(())
    }
}

/// One action-conditioned `64x64` probability map.
#[derive(Clone, Debug, PartialEq)]
pub struct ChangePrediction {
    pub action: ArcAction,
    /// Row-major probability that the corresponding visible pixel changes.
    pub probabilities: Vec<f32>,
}

/// Thresholded pre-update diagnostics for one confirmed transition.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectMetrics {
    pub balanced_bce: f64,
    pub changed_pixels: usize,
    pub unchanged_pixels: usize,
    pub predicted_changed_pixels: usize,
    pub true_positive: usize,
    pub true_negative: usize,
    pub false_positive: usize,
    pub false_negative: usize,
    /// `None` when the prediction contains no positive pixels.
    pub precision: Option<f64>,
    /// `None` when the factual transition contains no changed pixels.
    pub recall: Option<f64>,
    pub accuracy: f64,
}

/// Diagnostics for the optimizer update caused by one observation.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectUpdate {
    pub update: u64,
    pub batch_size: usize,
    pub loss: f64,
    pub changed_pixels: usize,
    pub unchanged_pixels: usize,
    pub changed_weight: f64,
    pub unchanged_weight: f64,
    pub gradient_l2: f64,
}

/// Cumulative factual and optimizer counts. These do not contain game identity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectTotals {
    pub observations: u64,
    pub optimizer_updates: u64,
    pub replay_len: usize,
    pub observed_changed_pixels: u64,
    pub observed_unchanged_pixels: u64,
}

/// Result of scoring and then learning from one confirmed factual pair.
#[derive(Clone, Debug, PartialEq)]
pub struct ObserveResult {
    /// Prediction made before this factual pair changed the parameters.
    pub pre_update: ChangePrediction,
    pub pre_update_metrics: EffectMetrics,
    pub update: EffectUpdate,
    pub totals: EffectTotals,
}

struct EffectNet {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    output: Conv2d,
}

impl EffectNet {
    fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        let spatial = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            c1: conv2d(INPUT_CHANNELS, hidden, 3, spatial, vb.pp("c1"))?,
            c2: conv2d(hidden, hidden, 3, spatial, vb.pp("c2"))?,
            c3: conv2d(hidden, hidden, 3, spatial, vb.pp("c3"))?,
            output: conv2d(hidden, 1, 1, Default::default(), vb.pp("output"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.c1.forward(input)?.relu()?;
        let hidden = self.c2.forward(&hidden)?.relu()?;
        let hidden = self.c3.forward(&hidden)?.relu()?;
        self.output.forward(&hidden).map_err(Into::into)
    }
}

#[derive(Clone)]
struct FactualTransition {
    current: ArcFrame,
    action: ArcAction,
    next: ArcFrame,
}

/// Fresh, bounded online visible-effect learner.
pub struct OnlineEffectLearner {
    device: Device,
    config: OnlineEffectConfig,
    model: EffectNet,
    parameters: Vec<(String, Var)>,
    optimizer: AdamW,
    replay: VecDeque<FactualTransition>,
    totals: EffectTotals,
}

impl OnlineEffectLearner {
    /// Construct a fresh learner with deterministic parameter values.
    ///
    /// The same seed produces identical initial weights on a given backend.
    /// Backend convolution implementations may still differ numerically.
    pub fn new(seed: u64, device: &Device, config: OnlineEffectConfig) -> Result<Self> {
        config.validate()?;
        let varmap = VarMap::new();
        let model = EffectNet::new(
            config.hidden_channels,
            VarBuilder::from_varmap(&varmap, DType::F32, device),
        )?;
        reinitialize_deterministically(&varmap, seed)?;
        let parameters = sorted_float_parameters(&varmap);
        ensure!(
            !parameters.is_empty(),
            "online effect model has no parameters"
        );
        ensure_parameters_finite(&parameters)?;
        let optimizer = AdamW::new(
            parameters.iter().map(|(_, var)| var.clone()).collect(),
            ParamsAdamW {
                lr: config.learning_rate,
                weight_decay: 0.0,
                ..ParamsAdamW::default()
            },
        )?;
        Ok(Self {
            device: device.clone(),
            config,
            model,
            parameters,
            optimizer,
            replay: VecDeque::with_capacity(config.replay_capacity),
            totals: EffectTotals::default(),
        })
    }

    pub fn config(&self) -> OnlineEffectConfig {
        self.config
    }

    pub fn totals(&self) -> &EffectTotals {
        &self.totals
    }

    /// Predict visible-change maps for several candidate actions on one frame.
    pub fn predict_batch(
        &self,
        frame: &ArcFrame,
        actions: &[ArcAction],
    ) -> Result<Vec<ChangePrediction>> {
        validate_frame(frame, "prediction frame")?;
        ensure!(
            !actions.is_empty(),
            "predict_batch requires at least one action"
        );
        for action in actions {
            validate_action(action)?;
        }
        let rows = actions
            .iter()
            .map(|action| (frame, action))
            .collect::<Vec<_>>();
        let input = encode_inputs(&rows, &self.device)?;
        let logits = self.model.forward(&input)?;
        probabilities_from_logits(&logits, actions)
    }

    /// Score, retain, and train once on a confirmed factual transition.
    ///
    /// The returned prediction and metrics are computed before the optimizer
    /// sees this pair. Replay sampling is deterministic: one update uses the
    /// most recent `physical_batch` retained transitions.
    pub fn observe(
        &mut self,
        current: &ArcFrame,
        action: &ArcAction,
        next: &ArcFrame,
    ) -> Result<ObserveResult> {
        validate_pair(current, action, next)?;

        let row = [(current, action)];
        let logits = self.model.forward(&encode_inputs(&row, &self.device)?)?;
        let probabilities = probabilities_from_logits(&logits, std::slice::from_ref(action))?
            .pop()
            .context("missing pre-update prediction")?;
        let target = change_targets(current, next);
        let pre_update_metrics = metrics_from(&logits, &probabilities.probabilities, &target)?;

        if self.replay.len() == self.config.replay_capacity {
            self.replay.pop_front();
        }
        self.replay.push_back(FactualTransition {
            current: current.clone(),
            action: action.clone(),
            next: next.clone(),
        });
        self.totals.observations += 1;
        self.totals.observed_changed_pixels += pre_update_metrics.changed_pixels as u64;
        self.totals.observed_unchanged_pixels += pre_update_metrics.unchanged_pixels as u64;

        let update = self.train_recent_replay()?;
        self.totals.optimizer_updates = update.update;
        self.totals.replay_len = self.replay.len();
        Ok(ObserveResult {
            pre_update: probabilities,
            pre_update_metrics,
            update,
            totals: self.totals.clone(),
        })
    }

    fn train_recent_replay(&mut self) -> Result<EffectUpdate> {
        let batch_size = self.replay.len().min(self.config.physical_batch);
        ensure!(batch_size > 0, "cannot train with empty replay");
        let start = self.replay.len() - batch_size;
        let batch = self.replay.iter().skip(start).collect::<Vec<_>>();
        let rows = batch
            .iter()
            .map(|sample| (&sample.current, &sample.action))
            .collect::<Vec<_>>();
        let targets = batch
            .iter()
            .flat_map(|sample| change_targets(&sample.current, &sample.next))
            .collect::<Vec<_>>();
        let logits = self.model.forward(&encode_inputs(&rows, &self.device)?)?;
        let target_tensor = Tensor::from_vec(
            targets.clone(),
            (batch_size, 1, FRAME_SIDE, FRAME_SIDE),
            &self.device,
        )?;
        let (loss, balance) = balanced_bce_with_logits(&logits, &target_tensor, &targets)?;
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        ensure!(
            loss_value.is_finite(),
            "online effect loss is non-finite: {loss_value}"
        );

        let grads = loss.backward()?;
        let gradient_l2 = finite_gradient_l2(&grads, &self.parameters)?;
        ensure!(
            gradient_l2 > 0.0,
            "online effect gradient norm is zero before update"
        );
        self.optimizer.step(&grads)?;
        ensure_parameters_finite(&self.parameters)?;

        Ok(EffectUpdate {
            update: self.totals.optimizer_updates + 1,
            batch_size,
            loss: loss_value,
            changed_pixels: balance.changed,
            unchanged_pixels: balance.unchanged,
            changed_weight: balance.changed_weight,
            unchanged_weight: balance.unchanged_weight,
            gradient_l2,
        })
    }
}

fn validate_frame(frame: &ArcFrame, label: &str) -> Result<()> {
    ensure!(
        frame.width as usize == FRAME_SIDE && frame.height as usize == FRAME_SIDE,
        "{label} must be {FRAME_SIDE}x{FRAME_SIDE}, got {}x{}",
        frame.width,
        frame.height
    );
    ensure!(
        frame.pixels.len() == FRAME_PIXELS,
        "{label} has {} pixels, expected {FRAME_PIXELS}",
        frame.pixels.len()
    );
    for (index, &pixel) in frame.pixels.iter().enumerate() {
        ensure!(
            pixel < 16,
            "{label} palette value {pixel} at index {index} is invalid"
        );
    }
    Ok(())
}

fn validate_action(action: &ArcAction) -> Result<()> {
    ensure!(
        (1..=7).contains(&action.id),
        "online effect actions must have id 1..=7; RESET/NULL is not trainable"
    );
    match action.id {
        6 => {
            let (x, y) = action
                .x
                .zip(action.y)
                .context("ACTION6 requires both coordinates")?;
            ensure!(x < 64 && y < 64, "ACTION6 coordinates must be in 0..64");
        }
        _ => ensure!(
            action.x.is_none() && action.y.is_none(),
            "coordinates are allowed only for ACTION6"
        ),
    }
    Ok(())
}

fn validate_pair(current: &ArcFrame, action: &ArcAction, next: &ArcFrame) -> Result<()> {
    validate_frame(current, "current frame")?;
    validate_frame(next, "next frame")?;
    validate_action(action)
}

fn encode_inputs(rows: &[(&ArcFrame, &ArcAction)], device: &Device) -> Result<Tensor> {
    ensure!(!rows.is_empty(), "cannot encode an empty effect batch");
    let mut values = vec![0f32; rows.len() * INPUT_CHANNELS * FRAME_PIXELS];
    for (batch, (frame, action)) in rows.iter().enumerate() {
        validate_frame(frame, "effect input frame")?;
        validate_action(action)?;
        let batch_offset = batch * INPUT_CHANNELS * FRAME_PIXELS;
        for (pixel_index, &palette) in frame.pixels.iter().enumerate() {
            values[batch_offset + usize::from(palette) * FRAME_PIXELS + pixel_index] = 1.0;
        }
        let action_plane = PALETTE_CHANNELS + usize::from(action.id - 1);
        let action_start = batch_offset + action_plane * FRAME_PIXELS;
        values[action_start..action_start + FRAME_PIXELS].fill(1.0);
        if action.id == 6 {
            let (x, y) = action
                .x
                .zip(action.y)
                .context("validated ACTION6 lost coordinates")?;
            let coordinate_plane = PALETTE_CHANNELS + ACTION_CHANNELS;
            values[batch_offset
                + coordinate_plane * FRAME_PIXELS
                + usize::from(y) * FRAME_SIDE
                + usize::from(x)] = 1.0;
        }
    }
    Tensor::from_vec(
        values,
        (rows.len(), INPUT_CHANNELS, FRAME_SIDE, FRAME_SIDE),
        device,
    )
    .map_err(Into::into)
}

fn change_targets(current: &ArcFrame, next: &ArcFrame) -> Vec<f32> {
    current
        .pixels
        .iter()
        .zip(next.pixels.iter())
        .map(|(before, after)| if before != after { 1.0 } else { 0.0 })
        .collect()
}

fn probabilities_from_logits(
    logits: &Tensor,
    actions: &[ArcAction],
) -> Result<Vec<ChangePrediction>> {
    let (batch, channels, height, width) = logits.dims4()?;
    ensure!(
        batch == actions.len() && channels == 1 && height == FRAME_SIDE && width == FRAME_SIDE,
        "effect logits shape must be Bx1x{FRAME_SIDE}x{FRAME_SIDE}, got {batch}x{channels}x{height}x{width}"
    );
    let flat = candle_nn::ops::sigmoid(&logits.detach())?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for (index, probability) in flat.iter().enumerate() {
        ensure!(
            probability.is_finite(),
            "effect probability at flat index {index} is non-finite"
        );
    }
    Ok(actions
        .iter()
        .cloned()
        .zip(flat.chunks_exact(FRAME_PIXELS))
        .map(|(action, probabilities)| ChangePrediction {
            action,
            probabilities: probabilities.to_vec(),
        })
        .collect())
}

#[derive(Clone, Copy)]
struct ClassBalance {
    changed: usize,
    unchanged: usize,
    changed_weight: f64,
    unchanged_weight: f64,
}

fn class_balance(targets: &[f32]) -> Result<ClassBalance> {
    ensure!(!targets.is_empty(), "effect targets must not be empty");
    let changed = targets.iter().filter(|&&target| target == 1.0).count();
    ensure!(
        targets.iter().all(|&target| target == 0.0 || target == 1.0),
        "effect targets must be binary"
    );
    let unchanged = targets.len() - changed;
    let (changed_weight, unchanged_weight) = match (changed, unchanged) {
        (0, _) => (0.0, 1.0),
        (_, 0) => (1.0, 0.0),
        _ => {
            let total = targets.len() as f64;
            (
                total / (2.0 * changed as f64),
                total / (2.0 * unchanged as f64),
            )
        }
    };
    Ok(ClassBalance {
        changed,
        unchanged,
        changed_weight,
        unchanged_weight,
    })
}

/// Saturation-safe elementwise BCE with smooth logits gradients.
///
/// With detached `m=max(-x,0)`, this evaluates
/// `x - x*t + m + log(exp(-m) + exp(-x-m))`. Detaching only the numerical
/// stabilizer leaves the analytical derivative exactly `sigmoid(x)-t`,
/// including `0.5-t` at `x=0`.
fn stable_bce_elements(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let max_negative = logits.neg()?.relu()?.detach();
    let xt = logits.broadcast_mul(targets)?;
    let first_exp = max_negative.neg()?.exp()?;
    let second_exp = logits.neg()?.sub(&max_negative)?.exp()?;
    let log_sum_exp = first_exp.add(&second_exp)?.log()?;
    logits
        .sub(&xt)?
        .add(&max_negative)?
        .add(&log_sum_exp)
        .map_err(Into::into)
}

fn balanced_bce_with_logits(
    logits: &Tensor,
    targets: &Tensor,
    host_targets: &[f32],
) -> Result<(Tensor, ClassBalance)> {
    ensure!(
        logits.dims() == targets.dims(),
        "effect logits/targets shape mismatch: {:?} vs {:?}",
        logits.dims(),
        targets.dims()
    );
    ensure!(
        logits.elem_count() == host_targets.len(),
        "host target count {} != tensor elements {}",
        host_targets.len(),
        logits.elem_count()
    );
    let balance = class_balance(host_targets)?;
    let weights = host_targets
        .iter()
        .map(|&target| {
            if target == 1.0 {
                balance.changed_weight as f32
            } else {
                balance.unchanged_weight as f32
            }
        })
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(weights, logits.shape(), logits.device())?;
    let weight_sum = balance.changed as f64 * balance.changed_weight
        + balance.unchanged as f64 * balance.unchanged_weight;
    ensure!(
        weight_sum > 0.0 && weight_sum.is_finite(),
        "invalid BCE weight sum"
    );
    let loss = stable_bce_elements(logits, targets)?
        .mul(&weights)?
        .sum_all()?
        .affine(1.0 / weight_sum, 0.0)?;
    Ok((loss, balance))
}

fn metrics_from(logits: &Tensor, probabilities: &[f32], targets: &[f32]) -> Result<EffectMetrics> {
    ensure!(
        probabilities.len() == targets.len(),
        "prediction/target length mismatch"
    );
    let target_tensor = Tensor::from_vec(
        targets.to_vec(),
        (1, 1, FRAME_SIDE, FRAME_SIDE),
        logits.device(),
    )?;
    let (loss, balance) = balanced_bce_with_logits(logits, &target_tensor, targets)?;
    let balanced_bce = loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
    ensure!(balanced_bce.is_finite(), "pre-update BCE is non-finite");

    let mut true_positive = 0;
    let mut true_negative = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;
    for (&probability, &target) in probabilities.iter().zip(targets) {
        ensure!(
            probability.is_finite(),
            "pre-update probability is non-finite"
        );
        match (probability >= 0.5, target == 1.0) {
            (true, true) => true_positive += 1,
            (false, false) => true_negative += 1,
            (true, false) => false_positive += 1,
            (false, true) => false_negative += 1,
        }
    }
    let predicted_changed_pixels = true_positive + false_positive;
    let precision = (predicted_changed_pixels > 0)
        .then(|| true_positive as f64 / predicted_changed_pixels as f64);
    let recall = (balance.changed > 0).then(|| true_positive as f64 / balance.changed as f64);
    Ok(EffectMetrics {
        balanced_bce,
        changed_pixels: balance.changed,
        unchanged_pixels: balance.unchanged,
        predicted_changed_pixels,
        true_positive,
        true_negative,
        false_positive,
        false_negative,
        precision,
        recall,
        accuracy: (true_positive + true_negative) as f64 / targets.len() as f64,
    })
}

fn sorted_float_parameters(varmap: &VarMap) -> Vec<(String, Var)> {
    let data = varmap.data().lock().expect("VarMap lock poisoned");
    let mut parameters = data
        .iter()
        .filter(|(_, var)| var.dtype().is_float())
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect::<Vec<_>>();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    parameters
}

fn stable_name_seed(master: u64, name: &str) -> u64 {
    name.as_bytes()
        .iter()
        .fold(master ^ 0x9E37_79B9_7F4A_7C15, |state, &byte| {
            state
                .wrapping_mul(0x0000_0100_0000_01B3)
                .wrapping_add(u64::from(byte))
        })
}

fn reinitialize_deterministically(varmap: &VarMap, master_seed: u64) -> Result<()> {
    let parameters = sorted_float_parameters(varmap);
    for (name, var) in parameters {
        let shape = var.shape().dims().to_vec();
        let values = if name.ends_with("bias") {
            vec![0.0; var.elem_count()]
        } else {
            let spatial = shape.iter().skip(2).product::<usize>().max(1);
            let fan_in = shape.get(1).copied().unwrap_or(1) * spatial;
            let fan_out = shape.first().copied().unwrap_or(1) * spatial;
            let bound = (6.0 / (fan_in + fan_out).max(1) as f64).sqrt() as f32;
            let mut rng = rand::rngs::StdRng::seed_from_u64(stable_name_seed(master_seed, &name));
            (0..var.elem_count())
                .map(|_| rng.random_range(-bound..=bound))
                .collect()
        };
        let tensor = Tensor::from_vec(values, shape.as_slice(), var.device())?;
        var.set(&tensor)
            .with_context(|| format!("initialize online effect parameter {name}"))?;
    }
    Ok(())
}

fn finite_gradient_l2(grads: &GradStore, parameters: &[(String, Var)]) -> Result<f64> {
    let mut sum = 0.0f64;
    for (name, var) in parameters {
        let gradient = grads
            .get(var.as_tensor())
            .with_context(|| format!("missing gradient for online effect parameter {name}"))?;
        for value in gradient
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
        {
            ensure!(value.is_finite(), "non-finite gradient in parameter {name}");
            sum += f64::from(value) * f64::from(value);
        }
    }
    let norm = sum.sqrt();
    ensure!(
        norm.is_finite(),
        "online effect gradient norm is non-finite"
    );
    Ok(norm)
}

fn ensure_parameters_finite(parameters: &[(String, Var)]) -> Result<()> {
    for (name, var) in parameters {
        for value in var
            .as_tensor()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
        {
            if !value.is_finite() {
                bail!("non-finite online effect parameter {name}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(replay_capacity: usize, physical_batch: usize) -> OnlineEffectConfig {
        OnlineEffectConfig {
            hidden_channels: 2,
            replay_capacity,
            physical_batch,
            learning_rate: 1e-3,
        }
    }

    fn frame(fill: u8) -> ArcFrame {
        ArcFrame::new(64, 64, vec![fill; FRAME_PIXELS]).unwrap()
    }

    fn parameter_bits(learner: &OnlineEffectLearner) -> Result<Vec<Vec<u32>>> {
        learner
            .parameters
            .iter()
            .map(|(_, var)| {
                Ok(var
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            })
            .collect()
    }

    #[test]
    fn seeded_initial_predictions_repeat_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let cfg = config(4, 2);
        let first = OnlineEffectLearner::new(42, &device, cfg)?;
        let second = OnlineEffectLearner::new(42, &device, cfg)?;
        let mut input = frame(0);
        input.pixels[9 * FRAME_SIDE + 7] = 3;
        let actions = [
            ArcAction::new(1, None, None)?,
            ArcAction::new(6, Some(7), Some(9))?,
        ];
        let left = first.predict_batch(&input, &actions)?;
        let right = second.predict_batch(&input, &actions)?;
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right.iter()) {
            assert_eq!(
                left.probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                right
                    .probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn encoding_distinguishes_action_ids_and_click_coordinates() -> Result<()> {
        let device = Device::Cpu;
        let input = frame(4);
        let actions = [
            ArcAction::new(1, None, None)?,
            ArcAction::new(2, None, None)?,
            ArcAction::new(6, Some(3), Some(5))?,
            ArcAction::new(6, Some(17), Some(29))?,
        ];
        let rows = actions
            .iter()
            .map(|action| (&input, action))
            .collect::<Vec<_>>();
        let encoded = encode_inputs(&rows, &device)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let at = |batch: usize, channel: usize, pixel: usize| {
            encoded[(batch * INPUT_CHANNELS + channel) * FRAME_PIXELS + pixel]
        };
        assert_eq!(at(0, PALETTE_CHANNELS, 0), 1.0);
        assert_eq!(at(0, PALETTE_CHANNELS + 1, 0), 0.0);
        assert_eq!(at(1, PALETTE_CHANNELS, 0), 0.0);
        assert_eq!(at(1, PALETTE_CHANNELS + 1, 0), 1.0);
        let coordinate = PALETTE_CHANNELS + ACTION_CHANNELS;
        assert_eq!(at(2, coordinate, 5 * FRAME_SIDE + 3), 1.0);
        assert_eq!(at(2, coordinate, 29 * FRAME_SIDE + 17), 0.0);
        assert_eq!(at(3, coordinate, 5 * FRAME_SIDE + 3), 0.0);
        assert_eq!(at(3, coordinate, 29 * FRAME_SIDE + 17), 1.0);
        Ok(())
    }

    #[test]
    fn balanced_loss_handles_absent_classes_and_saturated_logits() -> Result<()> {
        let device = Device::Cpu;
        for (targets, expected_weights) in [(vec![0.0; 4], (0.0, 1.0)), (vec![1.0; 4], (1.0, 0.0))]
        {
            let logits =
                Tensor::from_vec(vec![-1_000.0f32, 1_000.0, -100.0, 100.0], (4,), &device)?;
            let target_tensor = Tensor::from_vec(targets.clone(), (4,), &device)?;
            let (loss, balance) = balanced_bce_with_logits(&logits, &target_tensor, &targets)?;
            assert_eq!(
                (balance.changed_weight, balance.unchanged_weight),
                expected_weights
            );
            assert!(loss.to_scalar::<f32>()?.is_finite());
        }
        Ok(())
    }

    #[test]
    fn stable_bce_derivative_matches_sigmoid_minus_target_at_zero_and_extremes() -> Result<()> {
        let device = Device::Cpu;
        let values = vec![-1_000.0f32, -1.0, 0.0, 1.0, 1_000.0];
        for label in [0.0f32, 1.0] {
            let logits = Var::from_vec(values.clone(), (values.len(),), &device)?;
            let targets = Tensor::from_vec(vec![label; values.len()], (values.len(),), &device)?;
            let loss = stable_bce_elements(logits.as_tensor(), &targets)?.sum_all()?;
            let gradients = loss.backward()?;
            let actual = gradients
                .get(logits.as_tensor())
                .context("missing test logit gradient")?
                .to_vec1::<f32>()?;
            for (index, (&logit, &gradient)) in values.iter().zip(actual.iter()).enumerate() {
                let sigmoid = if logit >= 0.0 {
                    1.0 / (1.0 + (-logit).exp())
                } else {
                    let exp = logit.exp();
                    exp / (1.0 + exp)
                };
                let expected = sigmoid - label;
                assert!(
                    (gradient - expected).abs() <= 1e-6,
                    "label={label} index={index} logit={logit}: gradient {gradient} != {expected}"
                );
            }
            assert_eq!(actual[2], 0.5 - label);
        }
        Ok(())
    }

    #[test]
    fn changed_and_unchanged_observations_update_finite_parameters() -> Result<()> {
        let device = Device::Cpu;
        let mut learner = OnlineEffectLearner::new(7, &device, config(4, 2))?;
        let current = frame(0);
        let unchanged = current.clone();
        let action = ArcAction::new(5, None, None)?;

        let before = parameter_bits(&learner)?;
        let no_op = learner.observe(&current, &action, &unchanged)?;
        let after_no_op = parameter_bits(&learner)?;
        assert_ne!(before, after_no_op);
        assert_eq!(no_op.pre_update_metrics.changed_pixels, 0);
        assert_eq!(no_op.update.changed_weight, 0.0);
        assert_eq!(no_op.update.unchanged_weight, 1.0);
        assert!(no_op.update.loss.is_finite());
        assert!(no_op.update.gradient_l2.is_finite());

        let mut changed = current.clone();
        changed.pixels[11 * FRAME_SIDE + 13] = 8;
        let effect = learner.observe(&current, &action, &changed)?;
        let after_effect = parameter_bits(&learner)?;
        assert_ne!(after_no_op, after_effect);
        assert_eq!(effect.pre_update_metrics.changed_pixels, 1);
        assert_eq!(effect.update.batch_size, 2);
        assert_eq!(effect.update.changed_pixels, 1);
        assert!(effect.update.loss.is_finite());
        assert!(effect.update.gradient_l2.is_finite());
        Ok(())
    }

    #[test]
    fn replay_is_bounded_and_invalid_shapes_or_actions_fail_closed() -> Result<()> {
        let device = Device::Cpu;
        let mut learner = OnlineEffectLearner::new(9, &device, config(2, 2))?;
        let current = frame(1);
        let action = ArcAction::new(7, None, None)?;
        for color in [2, 3, 4] {
            let mut next = current.clone();
            next.pixels[usize::from(color)] = color;
            learner.observe(&current, &action, &next)?;
        }
        assert_eq!(learner.totals().observations, 3);
        assert_eq!(learner.totals().optimizer_updates, 3);
        assert_eq!(learner.totals().replay_len, 2);

        assert!(learner.predict_batch(&current, &[]).is_err());
        assert!(learner
            .predict_batch(&current, &[ArcAction::new(0, None, None)?])
            .is_err());
        let malformed_click = ArcAction {
            id: 6,
            x: Some(1),
            y: None,
        };
        assert!(learner.predict_batch(&current, &[malformed_click]).is_err());
        let wrong_shape = ArcFrame::new(63, 64, vec![0; 63 * 64])?;
        assert!(learner
            .predict_batch(&wrong_shape, &[action.clone()])
            .is_err());
        assert!(learner.observe(&current, &action, &wrong_shape).is_err());
        let mut nonfinite_logits = vec![0.0f32; FRAME_PIXELS];
        nonfinite_logits[17] = f32::NAN;
        let nonfinite_logits =
            Tensor::from_vec(nonfinite_logits, (1, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
        assert!(
            probabilities_from_logits(&nonfinite_logits, std::slice::from_ref(&action)).is_err()
        );
        assert!(OnlineEffectLearner::new(1, &device, config(1, 2)).is_err());
        Ok(())
    }
}
