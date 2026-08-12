//! One resolved identity for model topology, regularization, factual learning, and persistence.

use anyhow::{bail, ensure, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const LEGACY_SCHEMA: &str = "legacy_p2_eval_compatible";
pub const WORLD_CORE_V2_SCHEMA: &str = "world_core_v2";
pub const WORLD_CORE_V3_SCHEMA: &str = "world_core_v3";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SigregStatistic {
    #[default]
    EppsPulley,
    Quantile,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SigregPopulation {
    #[default]
    Marginal,
    TemporalResidual,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldCoreFamily {
    #[default]
    Legacy,
    V2,
    V3,
}

impl WorldCoreFamily {
    pub fn is_action_faithful(self) -> bool {
        !matches!(self, Self::Legacy)
    }

    pub fn schema(self) -> &'static str {
        match self {
            Self::Legacy => LEGACY_SCHEMA,
            Self::V2 => WORLD_CORE_V2_SCHEMA,
            Self::V3 => WORLD_CORE_V3_SCHEMA,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionConditioning {
    #[default]
    Global,
    SpatialField,
    SpatialResidual {
        scale: f64,
    },
}

impl ActionConditioning {
    pub fn uses_spatial_field(self) -> bool {
        !matches!(self, Self::Global)
    }

    pub fn residual_scale(self) -> Option<f64> {
        match self {
            Self::SpatialResidual { scale } => Some(scale),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SigregDefinition {
    pub enabled: bool,
    pub statistic: SigregStatistic,
    pub population: SigregPopulation,
    pub temporal_window: usize,
    pub global_mix: f64,
    pub spatial: bool,
    pub spatial_pool: bool,
    pub pre_rms_spatial: bool,
    pub legacy_loss_projector: bool,
    pub projector_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedExperiment {
    pub family: WorldCoreFamily,
    pub action_conditioning: ActionConditioning,
    pub sigreg: SigregDefinition,
    pub factual_learning: bool,
    pub report_schema: String,
}

impl Default for ResolvedExperiment {
    fn default() -> Self {
        Self {
            family: WorldCoreFamily::Legacy,
            action_conditioning: ActionConditioning::Global,
            sigreg: SigregDefinition {
                enabled: false,
                statistic: SigregStatistic::EppsPulley,
                population: SigregPopulation::Marginal,
                temporal_window: 8,
                global_mix: 0.0,
                spatial: false,
                spatial_pool: true,
                pre_rms_spatial: false,
                legacy_loss_projector: false,
                projector_dim: 128,
            },
            factual_learning: false,
            report_schema: LEGACY_SCHEMA.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExperimentRequest<'a> {
    pub world_core_v2: bool,
    pub world_core_v3: bool,
    pub spatial_action_field: bool,
    pub spatial_action_residual: bool,
    pub spatial_action_residual_scale: f64,
    pub branch_learning_enabled: bool,
    pub displacement_health_enabled: bool,
    pub sigreg_weight: f64,
    pub sigreg_statistic: SigregStatistic,
    pub sigreg_population: SigregPopulation,
    pub sigreg_temporal_window: usize,
    pub sigreg_global_mix: f64,
    pub sigreg_spatial: bool,
    pub sigreg_spatial_pool: bool,
    pub sigreg_pre_rms_spatial: bool,
    pub sigreg_projector: bool,
    pub sigreg_projector_dim: usize,
    pub lessons: &'a [String],
}

impl ResolvedExperiment {
    pub fn resolve(request: ExperimentRequest<'_>) -> Result<Self> {
        let family = match (request.world_core_v2, request.world_core_v3) {
            (false, false) => WorldCoreFamily::Legacy,
            (true, false) => WorldCoreFamily::V2,
            (true, true) => WorldCoreFamily::V3,
            (false, true) => bail!("world_core_v3 requires the world_core_v2 base topology"),
        };
        let factual_learning = family.is_action_faithful();
        ensure!(
            request.branch_learning_enabled == factual_learning,
            "resolved world-core family and branch_learning.enabled must match"
        );
        if factual_learning {
            ensure!(
                request
                    .lessons
                    .iter()
                    .any(|lesson| lesson == "factual_branches"),
                "action-faithful training requires a factual_branches lesson"
            );
            ensure!(
                request.sigreg_weight == 0.0,
                "V2/V3 use Consumer Latent health; set sigreg_weight=0"
            );
            ensure!(
                !request.sigreg_projector,
                "V2/V3 forbid the loss-only SIGReg projector"
            );
        }
        if request.displacement_health_enabled {
            ensure!(
                family == WorldCoreFamily::V3,
                "displacement health requires world-core-v3"
            );
        }

        let action_conditioning = match (
            request.spatial_action_field,
            request.spatial_action_residual,
        ) {
            (false, false) => ActionConditioning::Global,
            (true, false) => {
                ensure!(factual_learning, "spatial action fields require V2/V3");
                ActionConditioning::SpatialField
            }
            (true, true) => {
                ensure!(
                    family == WorldCoreFamily::V3,
                    "spatial action residual requires world-core-v3"
                );
                ensure!(
                    request.spatial_action_residual_scale.is_finite()
                        && (0.0..=1.0).contains(&request.spatial_action_residual_scale)
                        && request.spatial_action_residual_scale > 0.0,
                    "spatial action residual scale must be finite and in (0,1]"
                );
                ActionConditioning::SpatialResidual {
                    scale: request.spatial_action_residual_scale,
                }
            }
            (false, true) => bail!("spatial action residual requires a spatial action field"),
        };

        ensure!(
            !request.sigreg_projector || request.sigreg_projector_dim >= 2,
            "SIGReg projector dimension must be >= 2"
        );
        if request.sigreg_population == SigregPopulation::TemporalResidual {
            ensure!(
                request.sigreg_temporal_window >= 2,
                "temporally centered SIGReg requires a window >= 2"
            );
            ensure!(
                request.sigreg_spatial
                    && !request.sigreg_pre_rms_spatial
                    && !request.sigreg_projector,
                "temporally centered SIGReg requires post-RMS spatial geometry without a projector"
            );
            ensure!(
                request.sigreg_global_mix.is_finite()
                    && (0.0..=1.0).contains(&request.sigreg_global_mix),
                "temporally centered global mix must be finite and in [0,1]"
            );
        } else {
            ensure!(
                request.sigreg_global_mix == 0.0,
                "global mix requires temporally centered SIGReg"
            );
        }
        if request.sigreg_pre_rms_spatial {
            ensure!(
                request.sigreg_spatial && !request.sigreg_spatial_pool && !request.sigreg_projector,
                "pre-RMS spatial SIGReg requires unpooled spatial geometry without a projector"
            );
        }

        Ok(Self {
            family,
            action_conditioning,
            sigreg: SigregDefinition {
                enabled: request.sigreg_weight > 0.0,
                statistic: request.sigreg_statistic,
                population: request.sigreg_population,
                temporal_window: request.sigreg_temporal_window,
                global_mix: request.sigreg_global_mix,
                spatial: request.sigreg_spatial,
                spatial_pool: request.sigreg_spatial_pool,
                pre_rms_spatial: request.sigreg_pre_rms_spatial,
                legacy_loss_projector: request.sigreg_projector,
                projector_dim: request.sigreg_projector_dim,
            },
            factual_learning,
            report_schema: family.schema().into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(lessons: &'a [String]) -> ExperimentRequest<'a> {
        ExperimentRequest {
            world_core_v2: false,
            world_core_v3: false,
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: 0.25,
            branch_learning_enabled: false,
            displacement_health_enabled: false,
            sigreg_weight: 0.003,
            sigreg_statistic: SigregStatistic::EppsPulley,
            sigreg_population: SigregPopulation::Marginal,
            sigreg_temporal_window: 8,
            sigreg_global_mix: 0.0,
            sigreg_spatial: true,
            sigreg_spatial_pool: true,
            sigreg_pre_rms_spatial: false,
            sigreg_projector: false,
            sigreg_projector_dim: 128,
            lessons,
        }
    }

    #[test]
    fn v2_and_v3_reject_projector_geometry() {
        let lessons = vec!["factual_branches".into()];
        for v3 in [false, true] {
            let mut request = request(&lessons);
            request.world_core_v2 = true;
            request.world_core_v3 = v3;
            request.branch_learning_enabled = true;
            request.sigreg_weight = 0.0;
            request.sigreg_projector = true;
            assert!(ResolvedExperiment::resolve(request).is_err());
        }
    }

    #[test]
    fn resolved_identity_round_trips() -> Result<()> {
        let lessons = vec!["dynamics".into()];
        let resolved = ResolvedExperiment::resolve(request(&lessons))?;
        let json = serde_json::to_string(&resolved)?;
        assert_eq!(resolved, serde_json::from_str(&json)?);
        Ok(())
    }
}
