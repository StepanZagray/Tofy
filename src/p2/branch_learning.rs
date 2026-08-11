//! Factual same-state branch objectives for world-core-v2.

use crate::p2::data::{BranchGroup, FactualActionBranch, TransitionSample};
use crate::p2::model::{pool_latent, WorldModel};
use crate::p2::representation::{vicreg_latent_health, VicRegConfig};
use anyhow::{ensure, Result};
use candle_core::{Tensor, D};
use serde::{Deserialize, Serialize};

pub const WORLD_CORE_V2_SCHEMA: &str = "world_core_v2";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchLearningConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_outcome_pull_weight")]
    pub outcome_pull_weight: f64,
    #[serde(default = "default_outcome_push_weight")]
    pub outcome_push_weight: f64,
    #[serde(default = "default_outcome_margin")]
    pub outcome_margin: f64,
    #[serde(default = "default_action_recovery_weight")]
    pub action_recovery_weight: f64,
    #[serde(default = "default_coordinate_recovery_weight")]
    pub coordinate_recovery_weight: f64,
    #[serde(default = "default_changed_margin_weight")]
    pub changed_margin_weight: f64,
    #[serde(default = "default_changed_margin")]
    pub changed_margin: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_health: Option<VicRegConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooled_health: Option<VicRegConfig>,
}

fn default_outcome_pull_weight() -> f64 {
    0.05
}
fn default_outcome_push_weight() -> f64 {
    0.05
}
fn default_outcome_margin() -> f64 {
    0.5
}
fn default_action_recovery_weight() -> f64 {
    0.05
}
fn default_coordinate_recovery_weight() -> f64 {
    0.05
}
fn default_changed_margin_weight() -> f64 {
    0.05
}
fn default_changed_margin() -> f64 {
    0.1
}

impl Default for BranchLearningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            outcome_pull_weight: default_outcome_pull_weight(),
            outcome_push_weight: default_outcome_push_weight(),
            outcome_margin: default_outcome_margin(),
            action_recovery_weight: default_action_recovery_weight(),
            coordinate_recovery_weight: default_coordinate_recovery_weight(),
            changed_margin_weight: default_changed_margin_weight(),
            changed_margin: default_changed_margin(),
            spatial_health: None,
            pooled_health: None,
        }
    }
}

impl BranchLearningConfig {
    pub fn validate(&self, grad_accum: usize) -> Result<()> {
        for (name, value) in [
            ("outcome_pull_weight", self.outcome_pull_weight),
            ("outcome_push_weight", self.outcome_push_weight),
            ("action_recovery_weight", self.action_recovery_weight),
            (
                "coordinate_recovery_weight",
                self.coordinate_recovery_weight,
            ),
            ("changed_margin_weight", self.changed_margin_weight),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and >= 0"
            );
        }
        ensure!(
            self.outcome_margin.is_finite() && self.outcome_margin > 0.0,
            "outcome_margin must be finite and > 0"
        );
        ensure!(
            self.changed_margin.is_finite() && self.changed_margin > 0.0,
            "changed_margin must be finite and > 0"
        );
        if self.spatial_health.is_some() || self.pooled_health.is_some() {
            ensure!(
                grad_accum == 1,
                "world-core-v2 representation health requires grad_accum=1 so the nonlinear population objective sees the full physical batch"
            );
        }
        if let Some(config) = self.spatial_health {
            config.validate()?;
        }
        if let Some(config) = self.pooled_health {
            config.validate()?;
        }
        Ok(())
    }

    pub fn any_health_enabled(&self) -> bool {
        self.spatial_health.is_some() || self.pooled_health.is_some()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BranchLearningAudit {
    pub groups: usize,
    pub branches: usize,
    pub changed_branches: usize,
    pub equivalent_pairs: usize,
    pub distinct_pairs: usize,
    pub action6_branches: usize,
    pub action_recovery_branches: usize,
    pub spatial_population_rows: usize,
    pub pooled_population_rows: usize,
}

#[derive(Debug, Clone)]
pub struct BranchLearningLoss {
    pub total: Tensor,
    pub outcome_pull: Tensor,
    pub outcome_push: Tensor,
    pub action_recovery: Tensor,
    pub coordinate_recovery: Tensor,
    pub changed_margin: Tensor,
    pub spatial_variance: Tensor,
    pub spatial_covariance: Tensor,
    pub pooled_variance: Tensor,
    pub pooled_covariance: Tensor,
    pub audit: BranchLearningAudit,
}

fn mean_or_zero(values: Vec<Tensor>, zero: &Tensor) -> Result<Tensor> {
    if values.is_empty() {
        return Ok(zero.clone());
    }
    Tensor::stack(&values.iter().collect::<Vec<_>>(), 0)?
        .mean_all()
        .map_err(Into::into)
}

fn index_rows(rows: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let index = Tensor::from_vec(indices.to_vec(), (indices.len(),), rows.device())?;
    rows.index_select(&index, 0).map_err(Into::into)
}

fn validated_groups(samples: &[TransitionSample]) -> Result<Vec<BranchGroup>> {
    ensure!(!samples.is_empty(), "factual branch batch is empty");
    let mut groups = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let source = &samples[start];
        let mut end = start + 1;
        while end < samples.len()
            && samples[end].seed == source.seed
            && samples[end].episode_id == source.episode_id
            && samples[end].current == source.current
        {
            end += 1;
        }
        let branches = samples[start..end]
            .iter()
            .cloned()
            .map(FactualActionBranch::try_from_transition)
            .collect::<Result<Vec<_>>>()?;
        groups.push(BranchGroup::try_new(branches)?);
        start = end;
    }
    Ok(groups)
}

/// Compute all world-core-v2 objectives on the exact consumer latents used by
/// recurrence and prediction. Branch relations are active only for the named
/// factual curriculum; representation health can be active on every lesson.
pub fn branch_learning_loss(
    model: &WorldModel,
    samples: &[TransitionSample],
    current: &Tensor,
    predicted: &Tensor,
    target: &Tensor,
    config: &BranchLearningConfig,
    factual_curriculum: bool,
) -> Result<BranchLearningLoss> {
    let zero = current.zeros_like()?.sum_all()?;
    if !config.enabled {
        return Ok(BranchLearningLoss {
            total: zero.clone(),
            outcome_pull: zero.clone(),
            outcome_push: zero.clone(),
            action_recovery: zero.clone(),
            coordinate_recovery: zero.clone(),
            changed_margin: zero.clone(),
            spatial_variance: zero.clone(),
            spatial_covariance: zero.clone(),
            pooled_variance: zero.clone(),
            pooled_covariance: zero,
            audit: BranchLearningAudit::default(),
        });
    }
    ensure!(
        current.dims() == predicted.dims() && current.dims() == target.dims(),
        "world-core-v2 consumer latent shapes must match"
    );
    ensure!(
        current.dim(0)? == samples.len(),
        "consumer latent batch does not match factual samples"
    );

    let spatial_population = Tensor::cat(&[current, predicted, target], 0)?;
    let pooled_population = pool_latent(&spatial_population)?;
    let mut audit = BranchLearningAudit::default();
    let (spatial_variance, spatial_covariance, spatial_total) =
        if let Some(health) = config.spatial_health {
            let loss = vicreg_latent_health(&spatial_population, health)?;
            audit.spatial_population_rows = loss.rows;
            (loss.variance, loss.covariance, loss.weighted_total)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };
    let (pooled_variance, pooled_covariance, pooled_total) =
        if let Some(health) = config.pooled_health {
            let loss = vicreg_latent_health(&pooled_population, health)?;
            audit.pooled_population_rows = loss.rows;
            (loss.variance, loss.covariance, loss.weighted_total)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };

    let mut outcome_pull = zero.clone();
    let mut outcome_push = zero.clone();
    let mut action_recovery = zero.clone();
    let mut coordinate_recovery = zero.clone();
    let mut changed_margin = zero.clone();

    if factual_curriculum {
        let groups = validated_groups(samples)?;
        let displacement = pool_latent(&predicted.sub(current)?)?;
        let mut pull_terms = Vec::new();
        let mut push_terms = Vec::new();
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();
        let mut action6 = Vec::new();
        let mut recoverable = Vec::new();
        let mut offset = 0usize;
        for group in &groups {
            let branches = group.branches();
            for (local, branch) in branches.iter().enumerate() {
                let global = offset + local;
                if branch.board_effect.changed {
                    changed.push(global as u32);
                } else {
                    unchanged.push(global as u32);
                }
                if branch.transition.action.id == 6 {
                    action6.push(global as u32);
                }
                let unique_changed_effect = branch.board_effect.changed
                    && branches
                        .iter()
                        .filter(|candidate| branch.outcome_equivalent(candidate))
                        .count()
                        == 1;
                if unique_changed_effect {
                    recoverable.push(global as u32);
                }
                for other in local + 1..branches.len() {
                    let left = displacement.narrow(0, global, 1)?;
                    let right = displacement.narrow(0, offset + other, 1)?;
                    let distance = left.sub(&right)?.sqr()?.mean_all()?.sqrt()?;
                    if branch.outcome_equivalent(&branches[other]) {
                        pull_terms.push(distance.sqr()?);
                        audit.equivalent_pairs += 1;
                    } else {
                        push_terms.push(distance.affine(-1.0, config.outcome_margin)?.relu()?);
                        audit.distinct_pairs += 1;
                    }
                }
            }
            offset += branches.len();
        }
        audit.groups = groups.len();
        audit.branches = samples.len();
        audit.changed_branches = changed.len();
        audit.action6_branches = action6.len();
        audit.action_recovery_branches = recoverable.len();
        outcome_pull = mean_or_zero(pull_terms, &zero)?;
        outcome_push = mean_or_zero(push_terms, &zero)?;

        let (action_logits, coordinate_prediction) =
            model.decode_action_displacement(&displacement)?;
        if !recoverable.is_empty() {
            let recoverable_logits = index_rows(&action_logits, &recoverable)?;
            let actions = Tensor::from_vec(
                recoverable
                    .iter()
                    .map(|&index| u32::from(samples[index as usize].action.id))
                    .collect::<Vec<_>>(),
                (recoverable.len(),),
                current.device(),
            )?;
            action_recovery = candle_nn::loss::cross_entropy(&recoverable_logits, &actions)?;
        }
        let recoverable_action6 = action6
            .into_iter()
            .filter(|index| recoverable.contains(index))
            .collect::<Vec<_>>();
        if !recoverable_action6.is_empty() {
            let predicted_coords = index_rows(&coordinate_prediction, &recoverable_action6)?;
            let expected_coords = Tensor::from_vec(
                recoverable_action6
                    .iter()
                    .flat_map(|&index| {
                        let action = &samples[index as usize].action;
                        [
                            f32::from(action.x.expect("ACTION6 x")) / 63.0,
                            f32::from(action.y.expect("ACTION6 y")) / 63.0,
                        ]
                    })
                    .collect::<Vec<_>>(),
                (recoverable_action6.len(), 2),
                current.device(),
            )?;
            coordinate_recovery = predicted_coords.sub(&expected_coords)?.sqr()?.mean_all()?;
        }

        let displacement_norm = displacement.sqr()?.mean(D::Minus1)?.sqrt()?;
        let changed_loss = if changed.is_empty() {
            zero.clone()
        } else {
            index_rows(&displacement_norm, &changed)?
                .affine(-1.0, config.changed_margin)?
                .relu()?
                .mean_all()?
        };
        let copy_loss = if unchanged.is_empty() {
            zero.clone()
        } else {
            index_rows(&displacement_norm, &unchanged)?.mean_all()?
        };
        changed_margin = changed_loss.add(&copy_loss)?;
    }

    let mut total = spatial_total.add(&pooled_total)?;
    for (weight, loss) in [
        (config.outcome_pull_weight, &outcome_pull),
        (config.outcome_push_weight, &outcome_push),
        (config.action_recovery_weight, &action_recovery),
        (config.coordinate_recovery_weight, &coordinate_recovery),
        (config.changed_margin_weight, &changed_margin),
    ] {
        if weight > 0.0 {
            total = total.add(&loss.affine(weight, 0.0)?)?;
        }
    }
    Ok(BranchLearningLoss {
        total,
        outcome_pull,
        outcome_push,
        action_recovery,
        coordinate_recovery,
        changed_margin,
        spatial_variance,
        spatial_covariance,
        pooled_variance,
        pooled_covariance,
        audit,
    })
}
