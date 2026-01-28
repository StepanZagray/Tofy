# Tofy

A transformer encoder-decoder chatbot implemented in Rust using the Candle ML framework. Tofy uses multi-head self-attention in the encoder and cross-attention in the decoder for sequence-to-sequence conversation generation.

## Architecture

- **Encoder**: Transformer with self-attention layers
- **Decoder**: Transformer with self-attention and cross-attention to encoder output
- **Training**: Full sequence training with teacher forcing
- **Generation**: Autoregressive text generation with temperature sampling and top-k filtering

## Project Structure

```
src/
├── main.rs                    # Entry point and training loop
├── model/                      # Model components
│   ├── attention.rs           # Multi-head attention, transformer blocks
│   ├── encoder.rs             # Transformer encoder
│   ├── decoder.rs             # Transformer decoder with cross-attention
│   ├── vocab.rs               # Vocabulary and tokenization
│   └── predictor.rs           # (legacy, unused)
├── chatbot/                    # Chatbot functionality
│   ├── chatbot.rs             # Chat loop and model loading
│   └── generation.rs          # Text generation and sampling
├── data/                       # Data handling
│   └── data.rs                # Data loading, batching, encoding
└── config/                     # Configuration
    └── config.rs              # CLI argument parsing
```

## Data Format

Tofy reads a plain text file where each line is a dialogue pair:

```
context<TAB>response
```

Example:
```
hello how are you\tim fine thanks
```

The format also accepts `context|||response`. If there is no separator, the line is split in half by token count (fallback mode).

## Cornell Movie Dialogs

This repo includes scripts to convert the Cornell Movie Dialogs corpus into the required format.

### Convert Cornell → Training Pairs

```bash
python3 scripts/prepare_cornell.py \
  --corpus-dir "cornell movie-dialogs" \
  --output "data/cornell_pairs.txt" \
  --min-tokens 2 \
  --lower
```

This produces `data/cornell_pairs.txt`, which the Rust trainer expects.

## Training

The program automatically uses **CUDA(0)** if available, otherwise falls back to CPU.

### Training Command

```bash
cargo run --release -- <data_path> [steps] [batch] [dim] [max_ctx] [max_tgt] [num_layers] [num_heads]
```

**Arguments:**
- `data_path`: Path to training data file (tab-separated pairs)
- `steps`: Number of training steps (default: 10000)
- `batch`: Batch size (default: 16)
- `dim`: Embedding dimension (default: 256)
- `max_ctx`: Maximum context length (default: 64)
- `max_tgt`: Maximum target length (default: 32)
- `num_layers`: Number of transformer layers (default: 4)
- `num_heads`: Number of attention heads (default: 4)

**Example:**

```bash
# Quick test (10k steps)
cargo run --release -- data/cornell_pairs.txt 10000 16 256 64 32 4 4

# Longer training (500k steps)
cargo run --release -- data/cornell_pairs.txt 500000 32 256 64 32 4 4

# Larger model (384 dim, 6 layers)
cargo run --release -- data/cornell_pairs.txt 200000 16 384 64 32 6 6
```

**Recommended settings for RTX 5060 8GB:**
- Small: `dim=256, layers=4, heads=4` (~15M params) - fast training
- Medium: `dim=384, layers=6, heads=6` (~50M params) - better quality

After training, Tofy is saved to `model.safetensors` and vocabulary to `vocab.txt`.

## Chatbot Mode

Run Tofy in interactive chatbot mode:

```bash
cargo run --release -- --chat <model_path> <vocab_path> <data_path> [dim] [max_ctx] [num_layers] [num_heads]
```

**Arguments:**
- `--chat`: Enable chatbot mode
- `model_path`: Path to saved model (`model.safetensors`)
- `vocab_path`: Path to saved vocabulary (`vocab.txt`)
- `data_path`: Training data path (for reference, not used in generation)
- `dim`: Model dimension (must match training)
- `max_ctx`: Max context length (must match training)
- `num_layers`: Number of layers (must match training)
- `num_heads`: Number of attention heads (must match training)

**Example:**

```bash
cargo run --release -- --chat model.safetensors vocab.txt data/cornell_pairs.txt 256 64 4 4
```

Tofy:
1. Encodes your input using the transformer encoder
2. Generates a response autoregressively using the decoder with cross-attention
3. Uses temperature sampling and top-k filtering for diverse outputs

Type `quit` or `exit` to end the chat.

## Model Details

### Architecture
- **Multi-head attention**: Uses Candle's built-in primitives (`nn::Linear`, `Tensor::matmul`, `ops::softmax`)
- **Positional encoding**: Sinusoidal positional encodings
- **Layer normalization**: Pre-norm architecture for stable training
- **Feed-forward**: 4x embedding dimension with GELU activation

### Training
- **Loss**: Cross-entropy over all target tokens (excluding padding)
- **Optimizer**: AdamW with learning rate 3e-4
- **Teacher forcing**: Target sequence shifted right for training

### Generation
- **Sampling**: Temperature scaling (default: 0.8) with top-k filtering (default: 40)
- **Autoregressive**: Generates tokens one at a time using previous tokens

## Dependencies

- `candle-core` (0.9) with CUDA support
- `candle-nn` (0.9) with CUDA support
- `anyhow` for error handling
- `rand` for sampling

## Notes

- Tofy uses full sequence training (all tokens), not just the first token
- Padding tokens are masked during loss calculation
- The decoder uses cross-attention to attend to the encoder output
- For best results, train Tofy for 100k-500k steps depending on dataset size
- Larger models (more layers/heads) generally produce better quality but require more training time
