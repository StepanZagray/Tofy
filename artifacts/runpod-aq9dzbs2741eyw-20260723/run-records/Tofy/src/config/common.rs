use candle_core::DType;

pub(crate) fn parse_train_dtype() -> DType {
    std::env::var("TOFY_TRAIN_DTYPE")
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "float16" | "fp16" => Some(DType::F16),
            "bf16" => Some(DType::BF16),
            "f32" | "float32" | "fp32" => Some(DType::F32),
            _ => None,
        })
        .unwrap_or(DType::F32)
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}
