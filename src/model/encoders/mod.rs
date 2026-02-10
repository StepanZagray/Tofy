pub mod online_encoder;
mod shared;
pub mod teacher_encoder;

pub use online_encoder::OnlineEncoder;
pub use teacher_encoder::{copy_matching_vars, ema_update_matching_vars, TeacherEncoder};
