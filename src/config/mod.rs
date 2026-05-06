pub mod latent;
pub mod world;

pub use latent::{LatentEvalConfig, LatentTrainConfig};
pub use world::{
    DecoderTrainConfig, HighWorldTrainConfig, OrchestratorTrainConfig, ServeConfig,
    WorldEvalConfig, WorldTrainConfig,
};
