//! One resolved identity for model topology, regularization, factual learning, and persistence.

use anyhow::{bail, ensure, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::p2::grounding::PatchGroundingMode;

pub const LEGACY_SCHEMA: &str = "legacy_p2_eval_compatible";
pub const WORLD_CORE_V2_SCHEMA: &str = "world_core_v2";
pub const WORLD_CORE_V3_SCHEMA: &str = "world_core_v3";
pub const WORLD_CORE_V4_SCHEMA: &str = "world_core_v4_full_training";
pub const WORLD_CORE_V5_SCHEMA: &str = "world_core_v5";

/// A persisted training recipe, separate from historical research switches.
/// `FullV4` is resolved before validation and cannot be composed with V2/V3.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TrainingRecipe {
    #[default]
    LegacyExperimental,
    FullV4,
    FoundationV2,
}

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

/// The learned planning heads consume one `B×C` summary of the final spatial
/// prediction. This identity is deliberately separate from recurrence and
/// representation-health topology: both remain spatial in every variant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ConsumerReadoutTopology {
    #[default]
    GlobalMean,
    SpatialQuery,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldCoreFamily {
    #[default]
    Legacy,
    V2,
    V3,
    V4,
    V5,
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
            Self::V4 => WORLD_CORE_V4_SCHEMA,
            Self::V5 => WORLD_CORE_V5_SCHEMA,
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
    #[serde(default)]
    pub recipe: TrainingRecipe,
    pub family: WorldCoreFamily,
    pub action_conditioning: ActionConditioning,
    #[serde(default)]
    pub consumer_readout: ConsumerReadoutTopology,
    pub sigreg: SigregDefinition,
    #[serde(default)]
    pub patch_grounding_weight: f64,
    #[serde(default)]
    pub patch_grounding_mode: PatchGroundingMode,
    #[serde(default)]
    pub exact_grounding_weight: f64,
    pub factual_learning: bool,
    pub report_schema: String,
}

impl Default for ResolvedExperiment {
    fn default() -> Self {
        Self {
            recipe: TrainingRecipe::LegacyExperimental,
            family: WorldCoreFamily::Legacy,
            action_conditioning: ActionConditioning::Global,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
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
            patch_grounding_weight: 0.0,
            patch_grounding_mode: PatchGroundingMode::Both,
            exact_grounding_weight: 0.0,
            factual_learning: false,
            report_schema: LEGACY_SCHEMA.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExperimentRequest<'a> {
    pub recipe: TrainingRecipe,
    pub world_core_v2: bool,
    pub world_core_v3: bool,
    pub world_core_v4: bool,
    pub spatial_action_field: bool,
    pub spatial_action_residual: bool,
    pub spatial_action_residual_scale: f64,
    pub consumer_readout: ConsumerReadoutTopology,
    pub branch_learning_enabled: bool,
    pub displacement_health_enabled: bool,
    pub sigreg_weight: f64,
    pub patch_grounding_weight: f64,
    pub patch_grounding_mode: PatchGroundingMode,
    pub exact_grounding_weight: f64,
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
        let topology_family = match (
            request.world_core_v2,
            request.world_core_v3,
            request.world_core_v4,
        ) {
            (false, false, false) => WorldCoreFamily::Legacy,
            (true, false, false) => WorldCoreFamily::V2,
            (true, true, false) => WorldCoreFamily::V3,
            (false, false, true) => WorldCoreFamily::V4,
            (false, true, false) => bail!("world_core_v3 requires the world_core_v2 base topology"),
            _ => bail!("world-core-v4 is an exclusive successor topology"),
        };
        let family = match request.recipe {
            TrainingRecipe::FullV4 => {
                ensure!(
                    topology_family == WorldCoreFamily::V4,
                    "full-v4 recipe requires the world-core-v4 topology"
                );
                WorldCoreFamily::V4
            }
            TrainingRecipe::FoundationV2 => {
                ensure!(
                    topology_family == WorldCoreFamily::V4,
                    "foundation-v2 recipe requires the exact-decoder topology"
                );
                WorldCoreFamily::V5
            }
            TrainingRecipe::LegacyExperimental => {
                ensure!(
                    topology_family != WorldCoreFamily::V4,
                    "world-core-v4 topology requires a fixed successor recipe"
                );
                topology_family
            }
        };
        let legacy_branch_learning = matches!(family, WorldCoreFamily::V2 | WorldCoreFamily::V3);
        let factual_learning = legacy_branch_learning || family == WorldCoreFamily::V5;
        ensure!(
            request.branch_learning_enabled == legacy_branch_learning,
            "legacy branch_learning.enabled must match the V2/V3 family"
        );
        if legacy_branch_learning {
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
                ensure!(
                    factual_learning || family == WorldCoreFamily::V4,
                    "spatial action fields require V2/V3/V4/V5"
                );
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
            recipe: request.recipe,
            family,
            action_conditioning,
            consumer_readout: request.consumer_readout,
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
            patch_grounding_weight: request.patch_grounding_weight,
            patch_grounding_mode: request.patch_grounding_mode,
            exact_grounding_weight: request.exact_grounding_weight,
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
            recipe: TrainingRecipe::LegacyExperimental,
            world_core_v2: false,
            world_core_v3: false,
            world_core_v4: false,
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: 0.25,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
            branch_learning_enabled: false,
            displacement_health_enabled: false,
            sigreg_weight: 0.003,
            patch_grounding_weight: 0.0,
            patch_grounding_mode: PatchGroundingMode::Both,
            exact_grounding_weight: 0.0,
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

    #[test]
    fn full_v4_resolves_as_an_exclusive_successor() -> Result<()> {
        let lessons = vec!["dynamics".into()];
        let mut request = request(&lessons);
        request.recipe = TrainingRecipe::FullV4;
        request.world_core_v4 = true;
        request.spatial_action_field = true;
        request.consumer_readout = ConsumerReadoutTopology::SpatialQuery;
        request.sigreg_weight = 0.1;
        request.exact_grounding_weight = 0.1;
        let resolved = ResolvedExperiment::resolve(request)?;
        assert_eq!(resolved.family, WorldCoreFamily::V4);
        assert!(!resolved.factual_learning);
        assert_eq!(resolved.report_schema, WORLD_CORE_V4_SCHEMA);

        request.world_core_v2 = true;
        assert!(ResolvedExperiment::resolve(request).is_err());
        Ok(())
    }
}
