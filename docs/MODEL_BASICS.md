# Model basics: predictor, layers, MLP (for engineers)

Short, code-anchored explanation of the JEPA predictor and related terms.

## 1. What is a “predictor” in this project?

**In plain terms:** The predictor is a small neural net that takes the encoder’s representation at the mask position and outputs a vector that should match the **embedding** of the missing word.

**Data flow:**

```
Input phrase:  "do you _ it there"  (one token is masked)
       ↓
Encoder       →  one vector per position (including at the mask)
       ↓
Predictor     →  one vector that should point in the same “direction” as the true word’s embedding
       ↓
Compare       →  cosine similarity with every word’s embedding → pick best match
```

So “predictor” here is not a Candle type; it’s the **role** of this block in the architecture: it predicts (in embedding space) what token was at the mask.

**In code:** `Predictor` is our own struct in `src/model/predictor.rs`. It’s built from Candle’s **primitives** (two `nn::Linear` layers). Candle does not provide a “Predictor” layer; it provides building blocks, and we compose them into a predictor.

---

## 2. What is a “layer”?

A **layer** is one step of computation: it has parameters (e.g. a matrix and a bias) and a rule for computing outputs from inputs.

**Linear layer** (what Candle calls `nn::linear`):

- **Math:** `y = x W^T + b`
  - `x`: input vector of size `in_features`
  - `W`: weight matrix of shape `(out_features, in_features)`
  - `b`: bias vector of size `out_features`
  - `y`: output vector of size `out_features`
- **In code:** one `nn::Linear` = one such transformation. So “one linear layer” = one matrix multiply + add.

When we say “2-layer MLP,” we mean: input → **layer 1** → activation → **layer 2** → output. So “layers” here are these stacked computations (each with its own parameters).

---

## 3. What is an MLP?

**MLP = Multi-Layer Perceptron.**

- **Perceptron:** historically, a single linear transformation + threshold. In practice we use **linear + activation**.
- **Multi-layer:** stack several of these: layer₁ → activation → layer₂ → … → output.

So an MLP is: **input → (linear → activation) → (linear → activation) → … → output.**

**In our predictor** (`src/model/predictor.rs`):

```rust
// fc1, fc2 are both nn::Linear (from Candle)
let h = self.fc1.forward(x)?.relu()?;   // layer 1: linear then ReLU
Ok(self.fc2.forward(&h)?)                 // layer 2: linear, no activation on output
```

So we have:

- **Layer 1:** `fc1` (linear) + ReLU.
- **Layer 2:** `fc2` (linear), output is the predictor’s prediction.

That’s a 2-layer MLP: `dim → dim → dim` with ReLU in between.

---

## 4. Why not use “predictor from Candle”?

Candle does **not** provide a type called “Predictor” or a ready-made “MLP” struct. It provides:

- **Primitives:** `nn::linear`, `nn::embedding`, `nn::layer_norm`, etc.
- **Convention:** you implement the `Module` trait and build your own structs that call these primitives.

So in this project:

- **We do use Candle:** both “layers” inside our predictor are Candle’s `nn::Linear` (`nn::linear(...)`).
- **We don’t use a Candle predictor** because there isn’t one: we define our own `Predictor` that composes two `nn::Linear` layers.

**Alternative:** we could have inlined the same thing in `main.rs`:

```rust
let fc1 = nn::linear(dim, dim, vb.pp("predictor").pp("fc1"))?;
let fc2 = nn::linear(dim, dim, vb.pp("predictor").pp("fc2"))?;
// ... and in the training loop:
let h = fc1.forward(&hidden_at_mask)?.relu()?;
let pred_embed = fc2.forward(&h)?;
```

Putting that into `Predictor` in `predictor.rs` is just better structure: same Candle layers, clearer module boundary and reuse.

---

## 5. Summary table

| Term        | Meaning in this project |
|------------|--------------------------|
| **Predictor** | Our small net (in `predictor.rs`) that maps encoder hidden state at mask → vector in embedding space. Not a Candle type; we build it from Candle primitives. |
| **Layer**     | One parameterized computation step (e.g. one `nn::Linear`). |
| **MLP**       | Stack of (linear + activation) layers. Our predictor is a 2-layer MLP: `fc1 → ReLU → fc2`. |
| **Candle’s role** | Provides `nn::linear`, `nn::embedding`, etc. We use those to build the encoder and the predictor. |

So: the **predictor** is the component that turns the encoder’s “context” representation into a vector we compare to word embeddings; it’s implemented as a **2-layer MLP** using Candle’s **linear layers** only.
