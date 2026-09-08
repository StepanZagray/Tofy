//! From-scratch looped-transformer control experiments; no language-model weights.
//!
//! The first environment is a synthetic prerequisite, not an ARC game.

pub mod model;
pub mod profile;
pub mod task;

pub const FRAME_SIDE: usize = 64;
pub const PATCH_SIDE: usize = 8;
pub const PATCH_PIXELS: usize = PATCH_SIDE * PATCH_SIDE;
pub const PATCH_COUNT: usize = (FRAME_SIDE / PATCH_SIDE) * (FRAME_SIDE / PATCH_SIDE);
pub const PALETTE: usize = 16;
pub const ACTIONS: usize = 4;
pub const SUPPORT_STEPS: usize = 3;
pub const OBSERVED_FRAMES: usize = SUPPORT_STEPS * 2 + 1;
pub const TOKENS: usize = OBSERVED_FRAMES * PATCH_COUNT;
/// before/after/current one-hot (3), actual action one-hot (4), patch x/y (2),
/// and chronological transition index (1). No hidden rule or goal index.
pub const META_DIM: usize = 10;
