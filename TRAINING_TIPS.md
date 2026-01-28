# Training Tips for Better Generation

## Current Issues

The decoder was **not trained** in your original setup - only encoder + predictor were trained. I've now added decoder training, but here are additional tips:

## Improvements Made

1. **Added decoder training** - Now trains decoder to predict tokens from target latent
2. **Top-k sampling** - Only samples from top 50 tokens (reduces randomness)
3. **Better temperature** - Using 0.7 for more focused generation

## Training Recommendations

### 1. Train Longer
```bash
# You did 100k steps - that's good! But try:
cargo run --release -- data/cornell_pairs.txt 200000 32 256 64 32
```

### 2. Use Better Hyperparameters
```bash
# Larger model, more context:
cargo run --release -- data/cornell_pairs.txt 200000 32 512 128 64
```

### 3. Lower Learning Rate (for fine-tuning)
The decoder training might need a separate, lower learning rate. Currently both use the same LR.

### 4. Architecture Improvements

The current decoder is very simple. For better generation, consider:

- **Add more layers** to the decoder
- **Use attention** (self-attention or cross-attention)
- **Add positional encodings** for sequence awareness
- **Use a proper transformer decoder** architecture

## Why Generation Might Still Be Poor

1. **Simple architecture** - MLP decoder is very limited
2. **Limited training** - Decoder only sees first token prediction
3. **No sequence modeling** - Decoder doesn't see previous tokens well
4. **Dataset quality** - Cornell movie dialogs are very specific

## Quick Wins

1. **Train longer** - More steps = better representations
2. **Increase model size** - `dim=512` instead of `128`
3. **More context** - `max_ctx=128` instead of `32`
4. **Better dataset** - Use DailyDialog or Persona-Chat instead

## Next Steps

If generation is still poor after training:

1. Check if decoder loss is decreasing (should see it in logs)
2. Try retrieval mode instead (find closest training example)
3. Consider using a pre-trained language model for the decoder
4. Add more sophisticated sampling (top-p, nucleus sampling)
