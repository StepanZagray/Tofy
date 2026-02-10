# How the model trains, data size, and model size

## 1. How training works (step by step)

Each **step** does the following:

1. **Sample a batch**
   - `make_latent_batch` picks `batch_size` random **phrases** from your file (with replacement).
   - For each phrase it:
     - Picks a **random token position** (any non-pad, non-mask token).
     - Remembers that token as the **target** (the “answer”).
     - Replaces it with `<mask>` (or sometimes a random token, or leaves it) so the model sees the phrase with one hole.
   - So each batch is: 16 “fill-in-the-blank” examples: (phrase with one `_`, position of `_`, correct word).

2. **Forward pass**
   - **Encoder:** run the full phrase (with mask) through the transformer → one vector per position.
   - Take the vector **at the mask position** → that’s the “context” representation.
   - **Predictor:** map that vector to another vector of the same size (dim).
   - **Target:** the embedding of the true token (from the encoder’s embedding table).

3. **Loss (what we optimize)**
   - **MSE on directions:** predict and target are L2-normalized to unit vectors; loss = mean squared error between them (so we maximize cosine similarity).
   - **Ranking:** among 1 correct + 32 random wrong tokens, the correct one should have the **highest** cosine with the prediction (cross-entropy on these “logits”). See “Why ranking uses other tokens” below.
   - **Auxiliary:** the encoder’s hidden state at the mask should already point toward the target embedding (again via cosine); small extra term (weight 0.1).

4. **Backward**
   - One optimizer step (AdamW) on the combined loss.

So in one sentence: **the model is trained so that, given a phrase with one mask, the vector at the mask (after encoder + predictor) points in the same direction as the embedding of the missing word.**

---

### How to read the training log (step N/Total loss cos_mean aux acc_neg lr)

| Column   | Meaning |
|----------|--------|
| **loss** | Total loss (MSE + ranking + 0.1×aux). Should trend down. |
| **cos_mean** | Average cosine between **pred** and **target** (1.0 = perfect alignment). **High (0.99+) = training is working**; the prediction points in the right direction in embedding space. |
| **aux** | Auxiliary loss (1 − cos between encoder hidden at mask and target). Often ~0.9–1.0. |
| **acc_neg** | % of batches where the **correct** token has the **highest** cosine among 1 correct + 32 random negatives. Random = 1/33 ≈ 3%. **50–75% by the end = good**; 90%+ is strong. This metric is noisy (batch size 16) so it jumps step to step. |
| **lr** | Learning rate (warmup then cosine decay). |

**“Estimated parameters: ~0M”** was a display bug (integer division for &lt;1M models); it’s fixed to show e.g. ~0.99M. **If cos_mean is 0.99+ and acc_neg improved from ~6% to 50–75%, training did work.** The real check is inference: run `--diff` and see if top-5 nearest tokens and cosine with your answer look good on held-out phrases.

### acc_neg and acc_ema — what they are

- **acc_neg** (accuracy vs negatives): For each batch item we have **one correct token** (the true masked word) and **32 random negative tokens** from the vocab. We compute cosine(pred, correct) and cosine(pred, neg_i) for each negative. We count the item as **correct** if the correct token has **strictly the highest** cosine among the 33 (1 correct + 32 negatives). **acc_neg** = (number of such correct items) / batch_size. So it answers: “Does the model rank the true token above 32 random others?” Random chance = 1/33 ≈ 3%. This is a **ranking accuracy** over a small subset of the vocab (33 tokens), not over the full vocab.

- **acc_ema** (exponential moving average of acc_neg): **acc_ema = 0.92 × acc_ema + 0.08 × acc_neg**, updated every log step. It smooths the noisy per-batch acc_neg so you see a trend; it’s also used for **best-model checkpointing** (save when acc_ema improves). So acc_ema is not a different “kind” of accuracy — it’s just a smoothed version of acc_neg for logging and for choosing the best checkpoint.

### Other ways to measure accuracy for this model

This model predicts a **direction** in embedding space (no softmax over vocab). So “accuracy” can be defined in several ways:

| Metric | What it measures | Used in this codebase? |
|--------|-------------------|-------------------------|
| **acc_neg** (ranking over 1+32) | % of samples where correct token has highest cosine among 1 correct + 32 random negatives. | **Yes** — training log, best checkpoint. |
| **cos_mean** | Mean cosine(pred, target). Measures how well the predicted direction aligns with the true token (continuous, not discrete). | **Yes** — training log. |
| **Top-1 over full vocab** | % of samples where the correct token is the **nearest** among all vocab tokens (argmax over 4k cosines). Expensive (full vocab per sample). | **No** (could add for eval). |
| **Top-k over full vocab** | % where correct token is in the k nearest. Inference already shows “Top-5 nearest”; you could define accuracy as % in top-1, top-5, top-10. | **Partially** — inference prints top-5 and **rank** of your answer. |
| **Rank of correct token** | Rank of the correct token when all vocab tokens are sorted by cosine with pred (1 = best). Inference prints “your answer rank: R/V”. | **Yes** — inference only. |
| **Cosine threshold** | % of samples where cos(pred, target) &gt; τ (e.g. τ = 0.9). Simple but doesn’t account for how crowded the embedding space is. | **No**. |

So in this codebase: **acc_neg** = ranking accuracy over 33 tokens (fast, matches the ranking loss); **acc_ema** = smoothed acc_neg for logging and best-model selection; **cos_mean** = continuous alignment; **rank** at inference = how the model does over the full vocab for a single query.

---

### Training log patterns that can mean something is wrong

| Pattern | What it often means | What to try |
|--------|----------------------|-------------|
| **acc_neg stays near 0.03 (1/33)** for many steps | Model not learning to beat random negatives. | Smaller model, more data, or check data/vocab (e.g. OOV, empty pairs). |
| **cos_mean stays &lt; 0.99** and doesn’t improve | Predictions not aligning with target direction. | Longer training, smaller model for data size, or slightly higher LR. |
| **loss goes down but cos_mean / acc_neg don’t** | One loss term (e.g. aux) dominating; ranking/MSE not improving. | Rebalance loss weights or check that target embeddings are used correctly. |
| **acc_neg and cos_mean are good, inference is bad** | **Train/inference mismatch.** Model learned on one setup, inference uses another. | See “Train/inference parity” below. |
| **acc_neg very noisy** (e.g. 0.25 → 0.75 → 0.31) | Normal with small batch (16); metric is per-batch. | Ignore single steps; look at trend over hundreds of steps. |
| **acc_ema plateaus early** (e.g. flat from ~40k to 160k) | LR has decayed too much; model stops improving on ranking-over-33. | Use a **constant-LR phase** (code now keeps base_lr flat for ~25% of post-warmup steps, then cosine decay). Or **higher MSE_WEIGHT** (e.g. 4.0) so direction improves even when acc_ema is flat; **smaller max_vocab** (e.g. 4000) to reduce embedding crowding. |
| **loss explodes or NaN** | LR too high or numerical issue. | Lower LR, gradient clipping if added. |

**Train/inference parity (important):**  
Inference must use the **same sequence length and layout** as training. If you trained with `max_ctx=48`, `max_tgt=24` (seq_len 72), then `--diff` must use seq_len 72 so that context/target split is 48+24. The code now derives context/target from the `max_seq` you pass to `--diff` (e.g. `128 72 2 4` → seq_len 72, split 48+24). **Before this fix, inference hardcoded 64+32=96, so the model saw a different sequence length at inference than at training** — a common cause of “training looks good, inference is wrong.”

**If the smaller model still doesn’t work at inference:**  
1. **Use the same seq_len at inference as training.** Run `--diff ... 128 72 2 4` when you trained with 48+24=72.  
2. **Check vocab.** Inference vocab must be the same as training (same `vocab_latent.txt` produced by that run). If the answer token isn’t in vocab, you get “answer token not in vocab”.  
3. **More data or slightly larger model.** With 3k phrases, 731k params can still underfit the full vocab; try more data or a bit more capacity (e.g. dim 192, 3 layers) and retrain.

---

### Why does ranking use “other tokens”? Aren’t we only comparing pred and answer in embedding space?

We **are** only comparing in embedding space. The “other tokens” in the ranking step are **not** fed into the encoder or predictor.

What actually happens:

1. **Model input** (encoder + predictor) is only the **masked phrase**. No negative tokens are ever given as input.
2. After the forward pass we have **pred** (one vector per batch item) and we already have **target embedding** = embedding table row for the correct token.
3. For ranking we pick **32 random token IDs** (wrong words). We do **not** run the encoder on them. We only do **embedding lookup**: `encoder.embed_tokens(neg_ids)` → 32 vectors from the **same embedding table** (same table we use for the target). So we get 32 “other” vectors in embedding space.
4. We form 33 **scores** = cos(pred, target), cos(pred, neg_1), …, cos(pred, neg_32). The loss says: “the correct class is index 0” (i.e. the target should have the highest cosine). Cross-entropy pushes the model so that cos(pred, target) is larger than cos(pred, neg_i) for all i.

So: **no extra tokens are given to the model.** The model only sees the masked phrase. The negatives are just **which rows of the embedding table** we compare pred against. It’s “among these 33 directions in embedding space, pred should be closest to the correct one.” That makes the objective harder than MSE alone (you must beat 32 wrong options, not just get close to the target), which often helps learning.

---

## 2. Is ~3000 phrases too small?

It depends on **model size**, not only on phrase count.

- With **3k phrases** and ~5–10 words per phrase, you have on the order of **tens of thousands** of (phrase, position, word) combinations. Each training step uses 16 of them (with replacement), so in 50k steps you see **800k** such examples — but they are **repeats** of the same 3k phrases with different mask positions.
- So effectively you have **limited diversity**: the model only ever sees those phrases. If the model is **large**, it has enough parameters to memorize or to fit noise, and **accuracy can stay “random”** (e.g. acc_neg near 1/33) because:
  - It’s underfitting the hard objective (predict exact token in embedding space), or
  - It’s overfitting spurious patterns, or
  - Optimization is hard (bad conditioning, LR, etc.).

So: **3k phrases is small for a big model.** It’s not “too small” in absolute terms; it becomes too small when the model has many more parameters than you have effective training signal.

---

## 3. Why accuracy can stay “random” after 50k steps

Typical causes:

1. **Model too big for the data**
   - Example: dim=512, 6 layers, vocab 4k → ~20M+ parameters vs. ~30k “effective” (phrase, position, word) combinations. That’s hundreds of parameters per example → easy to overfit or fail to learn a clean mapping.
2. **Task difficulty**
   - Predicting a **single token in embedding space** (and then matching by cosine to 4k words) is hard; the predictor must map contextual representation → exact direction. Small errors in angle give wrong nearest neighbor.
3. **Optimization**
   - Learning rate, schedule, or batch size might be off; or the combination of MSE + ranking + auxiliary might need tuning.

So “50k steps and still random” usually means **model capacity and data size are mismatched** (often model too large), and/or the training setup needs adjustment.

---

### Flat 45–60% training accuracy from step 5k to 50k — is that overfitting?

**No.** That’s **not** overfitting.

- **Overfitting** = training accuracy keeps going **up** while the model memorizes; validation/test would get worse. You’d see training acc climbing (e.g. 60% → 80% → 95%) and staying high.
- **What you have** = training accuracy **plateaus** at 45–60% and doesn’t change for 45k steps. So the model **isn’t** memorizing; it learned something (45–60% is way above random 1/33 ≈ 3%) and then **stopped improving**. That’s **underfitting** or **stuck optimization**: the model isn’t good enough at the task even on the training set.

Possible causes and things to try:

1. **Learning rate decay too fast**  
   The schedule does warmup then cosine decay (3e-4 → 1e-5). By the second half of training the LR is small, so the model may be stuck in a bad region and can’t move. **Try:** keep LR constant for longer (e.g. 1e-4 or 3e-4 for 20k steps) or use a slower decay so LR stays higher in the 10k–40k range.

2. **Ceiling for this setup**  
   The task (predict exact token in embedding space, beat 32 random negatives) might be hard enough that this architecture/data tops out around 45–60% on this metric. **Try:** smaller model (sometimes optimizes better and generalizes better), or more data.

3. **Check cos_mean**  
   If both **acc_neg** and **cos_mean** (average cosine between pred and target) are flat from 5k onward, the model really stopped improving. If cos_mean is still creeping up while acc_neg is flat, the prediction is getting closer to the target but still often loses to one of the 32 negatives (e.g. unlucky negatives).

4. **Smaller model**  
   A smaller model (e.g. dim 128, 2 layers) often reaches a **higher** training accuracy on limited data because it has fewer bad local minima and optimizes more cleanly. Worth trying.

### Metrics vary but no trend after ~10k steps

If **loss**, **cos_mean**, and **acc_neg** bounce around and never improve (no downward/upward trend), common causes:

1. **Ranking loss too flat**  
   When cosines (pred vs target and vs negatives) are all in a narrow range (e.g. 0.9–1.0), the softmax over logits is very flat and gradients are weak. The code uses a **temperature** on the ranking logits (`RANK_TEMP = 0.1`): logits are scaled by `1/temperature` so the softmax is sharper and gradients are stronger. If you still see no trend, try a smaller temperature (e.g. 0.05) or a longer constant-LR phase.

2. **Learning rate**  
   Try keeping LR constant (e.g. 1e-4) for 20k steps before decay, or a slower decay, so the model has time to move.

3. **Loss balance**  
   If one term (MSE, ranking, or auxiliary) dominates, the model may optimize that term only. Check printed loss components; if needed, reweight (e.g. higher MSE weight to prioritize direction).

### acc_neg jumps a lot (e.g. 0.4 → 0.9 → 0.5)

This is **normal** with batch size 16 and 32 random negatives per sample: each batch has different phrases and different negatives, so acc_neg is noisy. The code now:

- Keeps an **EMA of acc_neg** (`acc_ema`, printed each log) to smooth the curve.
- **Saves the best model** whenever `acc_ema` improves (not the last step). So if you hit 0.93 at step 38900 and later steps are worse, `model_latent.safetensors` is the checkpoint from the best step, and inference uses that.

So even when the final step has low acc_neg, run inference with the saved file — it is the best checkpoint, not the last.

### Training cos_mean high (0.99+) but inference cosine low (~0.07)

Training reports **cos_mean** (average cosine over the batch). Inference reports cosine between **one** prediction and the answer token. Possible reasons the gap is large:

1. **Single example vs average**  
   cos_mean is an average over 16 samples; a few hard examples can have low cosine but not dominate the mean. Inference is one phrase at a time, so you see the hard cases.

2. **Best checkpoint**  
   Use the best saved model (see above). If the last step regressed, the old “final” save was worse; now the file is the best by acc_ema.

3. **Context frequency**  
   Phrases like “how _ you” or “do you _ anything” might appear rarely (one line each), so the model has seen them few times and may not generalize well.

4. **Direction vs nearest neighbor**  
   The model is trained to point in the right direction (high cosine). The vocab has 4k tokens; the nearest neighbor in embedding space can be a different word (e.g. phonetically similar) if the embedding manifold is crowded.

### When acc_ema is good (e.g. 0.71) but inference is still bad (cosine ~0.02–0.04, rank 1000+/4000)

**acc_ema** measures “beat 32 random negatives” (ranking over 33 tokens). Inference uses the **full vocab** (e.g. 4000 tokens). So **acc_ema 0.71 does not imply good full-vocab ranking**: the model can rank the correct token above 32 random others most of the time but still be far from “nearest among 4000” (e.g. rank 1140/4000, cosine 0.03).

**Causes and what to try:**

1. **Ranking over 33 vs over 4000**  
   Training only sees 32 random negatives per sample; the full vocab has 4000. The model may learn to beat random negatives without pointing precisely at the target. **Try:** increase **MSE weight** (e.g. `loss = 2.0 * mse + rank + 0.05 * aux`) so the objective pushes pred **exactly** toward the target direction, not just “above 32 others.”

2. **Embedding crowding**  
   4000 tokens in 240 dims → many words in similar directions; small angular errors give wrong nearest neighbor. **Try:** smaller **max_vocab** (e.g. 2000–3000) so the embedding space is less crowded, or slightly larger **dim** (e.g. 320) for the same vocab.

3. **Rare or unseen contexts**  
   Phrases like “Wow, what an overrated _ this turned out to be” may appear rarely (or never) in training; the model hasn’t learned that context. **Try:** more data, or check that your test phrases are in-domain and appear (or similar) in the training set.

4. **More steps / better alignment**  
   acc_ema 0.71 might still leave cosine with target modest on average. **Try:** longer training, or a **constant-LR phase** (e.g. 20k steps at 1e-4) before decay so the model keeps improving direction (cos_mean).

5. **Full-vocab eval during training**  
   To see if full-vocab ranking improves, you could add an **eval set** and periodically compute “% where correct token is in top-10” over the full vocab (expensive: one forward + 4000 cosines per sample). That would tell you whether the gap is “acc_neg is misleading” or “model needs more training / different objective.”

**Summary:** acc_ema measures a **weaker** task (beat 32 random) than inference (nearest among 4000). To improve inference, prioritize **direction** (higher MSE weight, cos_mean) and/or **less crowding** (smaller vocab or larger dim), and more data or steps if contexts are rare.

### When inference is still mediocre after flat-LR + MSE 4.0 (e.g. rank 1000+/8000, cosine ~0.05)

If you already use a constant-LR phase and MSE_WEIGHT 4.0 but full-vocab rank and cosine at inference stay poor:

1. **Smaller max_vocab**  
   8000 tokens in 384 dims → crowded embedding space; small angular errors → wrong nearest neighbor. **Retrain with max_vocab 4000–5000** (e.g. `... 5 6 5000`) so the same dim has fewer tokens and better angular separation.

2. **Higher MSE_WEIGHT**  
   In `main.rs` set **MSE_WEIGHT** to **6.0 or 8.0** so the loss pushes the prediction even more strongly toward the target direction. acc_neg may drop slightly but full-vocab rank often improves.

3. **Sharper ranking temperature**  
   In `main.rs` set **RANK_TEMP** from **0.1 to 0.05** so the softmax over (correct, 32 negatives) is sharper and gradients to separate correct from negatives are stronger.

4. **More data / steps**  
   If your test phrases are out-of-domain or rare, more data or longer training can help; but if cos_mean is already 0.99+ and acc_ema is good, the main lever is usually **crowding** (smaller vocab) and **direction** (higher MSE, lower RANK_TEMP).

---

## 4. How big should the model be for this task?

Rough rule of thumb: **number of parameters should be on the order of (or less than) the number of “effective” training examples**, or at least not 10–100× larger.

- **Effective examples:** e.g. 3000 phrases × ~8 words ≈ **24k** (phrase, position, target) combinations.
- So you want a **small** model: on the order of **1M parameters or less** for 3k phrases.

Example **small** config that fits the current code (and is more in line with 3k phrases):

- `dim`: **128** (instead of 512)
- `num_layers`: **2** (instead of 6)
- `num_heads`: **4**
- `max_vocab`: **4000** (or whatever you use)
- `max_ctx` / `max_tgt`: e.g. **48 / 24** (or 64/32)

Rough parameter count for that:

- Embedding: vocab_size × 128 ≈ 0.5M
- Transformer: 2 layers × (attention + FF) ≈ 0.4M
- Predictor: 2 × (128² + 128) ≈ 33k  
→ **~1M parameters** total.

Then 50k steps × 16 batch = 800k sample steps over ~24k unique patterns → many passes over the data, and the model has a chance to learn without huge overparameterization.

If you **increase data** (e.g. 50k–100k phrases), you can then scale up (e.g. dim 256, 4 layers) and expect better accuracy.

---

## 5. Does Candle provide better / bigger datasets? (candle-datasets)

**Yes.** The **candle-datasets** crate exists and provides:

- **Modules:** `batcher`, `hub`, `nlp`, `vision`
- **NLP:** includes at least **tinystories** (text, pre-tokenized for small LMs)
- **Hub:** integration with Hugging Face Hub, so you can pull standard datasets (e.g. more dialogue or text) if they’re in a supported format.

So you **can** use candle-datasets to:

1. **Use a bigger built-in dataset** (e.g. TinyStories) to get many more “phrases” (e.g. sliding windows or full sentences).
2. **Use the Hub** to load larger text or dialogue datasets and then treat each line (or each segment) as a “phrase” for your current pipeline.

Your current pipeline expects a **text file** (lines = phrases, optionally with tab/`|||`). To use candle-datasets you’d add a **data loader** that:

- Calls candle-datasets (e.g. `nlp::tinystories` or Hub) to get text.
- Converts that text into the same format your code expects (list of tokenized phrases / pairs), or adapts `build_vocab_and_pairs` / `make_latent_batch` to consume an iterator from candle-datasets instead of a single file.

So: **candle-datasets can give you larger/better datasets**; to use them you’d wire one of its data sources into your existing training loop (same loss and model, different data source).

---

### Parquet rows, local file, and HF cache

When you use `hub:<dataset_id>` (e.g. `hub:imdb`), the following applies.

- **Parquet rows**: Hugging Face stores datasets as **Parquet** (columnar table format). Think of it as a table: each **row** is one record (e.g. one review, one dialogue). Columns might be `text`, `label`, `dialog`, etc. “Parquet rows” = those table rows; we read them one by one and pull out the text column(s).

- **We do save the dataset locally**: The dataset is written to **`data/cached_<id>.txt`** (one phrase per line). That’s your local copy. Every later training run reads from this file; we don’t re-download or re-parse Parquet.

- **How HF caches**: The **hf-hub** crate (same idea as Python’s `huggingface_hub`) keeps a **download cache** on disk, usually `~/.cache/huggingface/hub/`. When we ask for a dataset, hf-hub downloads the Parquet files into that cache (and reuses them on later runs). So:
  1. **First run with `hub:...`**: hf-hub downloads Parquet into `~/.cache/...` (if not already there) → we read those Parquet files once, convert to text → write **`data/cached_<id>.txt`** → training uses that file.
  2. **Next runs**: We see **`data/cached_<id>.txt`** exists → we use it directly (no hf-hub call, no Parquet, no network).

So there are two levels: **hf-hub’s cache** (raw Parquet in `~/.cache/...`) and **our cache** (`data/cached_<id>.txt`). We use our cache for training so we never touch the network or Parquet after the first run.

---

## 6. Model size vs data (when inference is still poor)

**Your setup:** ~3.6k phrases, dim 240, 3 layers, 4 heads → **~2.76M params**. Inference cosine ~0.07–0.09, top-5 unrelated.

**Rule of thumb:** params should be on the order of (or less than) **effective examples**. Effective examples ≈ phrases × ~8 words (e.g. 3.6k × 8 ≈ **29k**). So 2.76M params is **~95 params per example** — not “too small”; for 3k phrases the doc recommends **~1M params** to avoid underfitting/instability.

| Data size | Suggested params | Example config |
|-----------|------------------|----------------|
| **~3k phrases** | **~1M** (smaller is often better) | dim 128, 2 layers, 4 heads → ~0.7–1M |
| **~3.6k phrases** (current) | **~1–1.5M** | dim 128–192, 2–3 layers → ~0.7–1.5M |
| **20k–50k phrases** | **2–4M** | dim 240–256, 3–4 layers → ~2–4M |
| **100k+ phrases** | **5M+** | dim 256–512, 4–6 layers |

So: **you don’t need a bigger model for 3.6k phrases** — you’re already above the recommended size. What usually helps:

1. **More data**  
   More phrases → more diverse (phrase, position, target) combinations → better generalization. Try 20k–50k phrases (e.g. more dialogue data, or candle-datasets) and then you can scale to 2–4M params.

2. **Smaller model for current data**  
   With 3.6k phrases, a **smaller** model (e.g. dim 128, 2 layers, ~1M params) often trains more stably and can give better inference than 2.76M params, because there’s less overparameterization and fewer bad local minima.

3. **Other problems that can cause poor inference**  
   - **Embedding crowding:** 4k tokens in 240 dims → many words in similar directions; nearest neighbor can be wrong even if direction is “close.” Try smaller vocab (e.g. top 2k by frequency) or slightly larger dim.  
   - **Rare contexts:** A phrase like “good luck _ with school” may appear once; the model rarely sees that exact context. More data helps.  
   - **Loss balance:** If ranking dominates, the model learns “beat 32 random negatives” but not “point exactly at target.” Try increasing MSE weight (e.g. `loss = 2.0 * mse + rank + 0.05 * aux`) so direction is prioritized.  
   - **Best checkpoint:** Ensure you’re using the best saved model (acc_ema), not the last step.  
   - **Evaluation:** Check the **rank** of your answer in the full vocab (e.g. “your answer rank: 234/4022”). If rank is in the hundreds or thousands, the model is only roughly in the right direction; if rank is in top-10, the setup is closer to working.

---

## 7. Dataset recommendations and model sizes by task

### Recommended datasets for this model (fill-in-the-blank in embedding space)

This model does **masked token prediction in embedding space**: given a phrase with one word masked, it predicts a direction that should align with the missing word's embedding. So you want **short phrases or utterances** (sentence-like, not long documents), with natural language and a vocab that matches your use case.

| Use case | Recommended dataset(s) | Size (phrases/lines) | Notes |
|----------|------------------------|----------------------|--------|
| **20–30M params (good balance)** | `hub:imdb`, `hub:ag_news`, `hub:yelp_polarity`, `hub:li2017dailydialog/daily_dialog` (flatten turns) | **50k–200k+** | See "Good datasets for 20–30M" below. |
| **General fill-in-the-blank** | `hub:imdb`, `hub:li2017dailydialog/daily_dialog` | 20k–50k+ | Good for a ~2–4M param model. |
| **Casual dialogue** | Your `data/casual_pairs.txt` (or `hub:...` dialogue datasets) | 3k → 20k+ | Start small (~1M params); scale data and model together. |
| **Simple / demo** | Same as above, smaller slice | 3k–10k | Use ~1M params; see §6 for "smaller is often better" on limited data. |

**Dataset size guideline for this model**

- **Effective examples** ≈ number of phrases × average words per phrase (e.g. 30k phrases × 8 words ≈ 240k (phrase, position, target) combinations).
- **Rule of thumb:** keep **model parameters on the order of (or less than) effective examples**. So:
  - **~30k effective examples** → ~1–3M params (e.g. dim 128–240, 2–3 layers).
  - **~200k–500k+ effective examples** → **~20–30M params** (good balance: more capacity than 3M, still trainable on moderate data).
  - **~200k+ effective examples** → ~2–10M params; scale up with more data.
- Too many params on too little data → unstable training, poor generalization. Too little data for a given size → underfitting or noisy metrics.

### How dim, layers, steps (and batch) are connected

**dim** = embedding dimension. It sets the size of each token vector and the width of the transformer. Larger **dim** → more capacity and more parameters (embedding table ≈ vocab × dim; each layer ≈ dim²). More **dim** also needs more data and more memory.

**layers** = number of transformer blocks (depth). More **layers** → deeper model, more capacity, more parameters (roughly linear in layers). Same idea: more layers needs enough data to train them.

**steps** = number of optimizer updates. Each step uses one **batch** of examples. So **total examples seen** = `steps × batch_size`. There is no fixed “epoch” in this setup (we sample with replacement), but you can think of **effective passes** over the data as:

- **effective_passes** ≈ (steps × batch_size) / effective_examples  

So: more **steps** or larger **batch** → the model sees the data more times. Too few steps → underfitting; too many steps with small data → can plateau or overfit.

**batch_size** = examples per step. Larger batch → fewer steps for the same number of examples seen, but each gradient is less noisy. Smaller batch → more steps, noisier gradients. Often 16–64; limited by GPU memory (batch × seq_len × dim).

**Rough tuning order**

1. **Data** — Fix your dataset (number of phrases/lines). Compute **effective_examples** ≈ phrases × ~words per phrase.
2. **Model size (dim, layers)** — Choose so **total params** are on the order of (or below) effective_examples (see table in §7). Example: 30k effective → ~1–3M params (dim 128–240, 2–3 layers); 500k effective → ~20–30M (dim 256–384, 4–6 layers).
3. **Steps** — Choose so the model sees the data several times. A simple target: **steps × batch_size ≥ 5–10 × effective_examples** (i.e. 5–10 “epochs” worth). So  
   **steps ≈ (5–10) × effective_examples / batch_size**.  
   Example: 240k effective examples, batch 16 → steps ≈ 75k–150k (e.g. 100k steps). For 50k phrases × 8 words = 400k effective, batch 32 → steps ≈ 62k–125k (e.g. 80k steps).
4. **batch_size** — As large as your GPU allows (sequence length × batch × dim and model size set memory). If you can’t increase steps much, slightly larger batch still increases total examples seen.
5. **max_ctx, max_tgt** — Match your typical phrase length (e.g. 48+24). Longer sequences = more memory per batch.

**Summary**

| You want… | Then… |
|------------|--------|
| **More capacity** | Increase **dim** and/or **layers** (and have enough data + steps). |
| **Model to see data more** | Increase **steps** and/or **batch_size**. |
| **Stable training** | Keep **params** on the order of **effective_examples**; use enough **steps** (e.g. 5–10× effective_examples / batch). |
| **Fit in GPU memory** | Lower **batch_size**, **dim**, **max_ctx**+**max_tgt**, or **layers**. |

### Batch size: memory and performance

**Memory** — Activations (hidden states) scale with **batch_size × seq_len × dim × layers**: each step stores a tensor of shape (batch, seq_len, dim) per layer. So **larger batch** → more memory per step. Optimizer state (e.g. Adam) depends on **params**, not batch, but the main GPU memory cost during training is usually activations, so **doubling batch_size** roughly doubles the activation memory for that step. If you hit OOM, reduce batch_size first (or seq_len, dim, layers).

**Performance (throughput)** — Larger batch often **improves samples per second** (better GPU utilization) up to a point: one step does more work, so time per step goes up, but usually not proportionally. Beyond some batch size you become memory- or compute-bound and gains flatten. So **larger batch** → often faster training in wall-clock time until you run out of memory.

**Performance (convergence / quality)** — Larger batch → **smoother gradients** (average over more examples), so each step is more stable but you get **fewer updates** for the same number of examples seen. Sometimes very large batches need more steps or a slightly higher LR to converge as well as smaller batches. **Smaller batch** → noisier gradients, more updates per "epoch"; can sometimes generalize slightly better but may need more steps. For this codebase, **batch 16–64** is a typical range; tune steps so that (steps × batch_size) still gives you 5–10× effective_examples.

### Good datasets for a 20–30M param model

**20–30M is a good balance** between minimal (1–3M) and full BERT-scale (60M+): enough capacity to learn better representations, but still trainable on hundreds of thousands of effective examples (e.g. 50k–200k+ phrases). Aim for **effective examples on the order of 20–30M or more** (e.g. 100k phrases × ~8 words ≈ 800k; or 50k phrases × ~10 words ≈ 500k).

| Dataset (Hub ID) | Approx. lines/phrases | Notes |
|------------------|------------------------|--------|
| **`hub:imdb`** | ~25k train + 25k test = **50k** | Movie reviews; one `text` per row. Good starting size for 20–30M. |
| **`hub:ag_news`** | **120k** train (news titles + descriptions) | One `text` per row; more data for better fit. |
| **`hub:yelp_polarity`** | **560k** (train) | Reviews; one `text` per row; large, good for 20–30M. |
| **`hub:li2017dailydialog/daily_dialog`** | ~11k dialogues → **~100k+ turns** when flattened | One turn per line (list column `dialog`); conversational. |
| **`hub:wikitext-2-raw-v1`** | ~36k articles (train) | One `text` per row (paragraph/article); can split by sentence for more lines. |

Use `hub:<dataset_id>` as `data_path`; the loader downloads once and caches under `data/cached_<id>.txt`. For 20–30M params, a config in the **dim 256–384, 4–6 layers** range is a rough fit (scale to your GPU and step count).

### AG News for a 12–25M param model (recommended example)

**AG News** (~120k train lines, one `text` per row) is a good fit for a **12–25M** param model: enough data to train that size, and the loader supports it via `hub:ag_news`.

**Example training command:**

```bash
cargo run --release -- --latent hub:ag_news 200000 32 256 48 24 4 8 16000
```

- **hub:ag_news** — dataset (downloaded once, cached as `data/cached_ag_news.txt`).
- **200000** steps × **32** batch → 6.4M samples seen; effective_examples ≈ 120k × ~10 words ≈ 1.2M, so ~5 "epochs" worth.
- **256** dim, **4** layers, **8** heads, **16000** max_vocab → **~12–17M params** (embedding + transformer + predictor). Tweak dim/layers to land in 12–25M (e.g. dim 280 or 5 layers for ~20M).
- **48 + 24** max_ctx + max_tgt (seq_len 72).

**Inference** (same dim, layers, heads, seq_len 72):

```bash
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 256 72 4 8
```

If you hit OOM, try **batch 16** (and optionally **150000** steps so you still see the data ~2×).

### Typical model sizes by task (reference)

Rough orders of magnitude from common practice (research and production). Your model is **small** (1–3M) and task-specific; these are for context.

| Task | Typical model size | Examples |
|------|--------------------|----------|
| **Fill-in-the-blank / masked LM** | **~100M–350M** (base), ~1B+ (large) | BERT-base ~110M, BERT-large ~340M; MLM pre-training then fine-tuning. |
| **Small / distilled fill-in-the-blank** | **~1M–30M** | *Distilled* = a smaller model trained to mimic a larger (teacher) model, e.g. DistilBERT ~66M from BERT-base; tiny BERTs 4–14M; **this codebase 1–3M** for a minimal latent-only setup. |
| **Conversation / dialogue** | **~100M–70B+** | Blender ~90M–9B; ChatGPT-style models 7B–70B+; small chit‑chat 100M–1B. |
| **Text classification** | **~1M–110M** | Linear on top of frozen BERT: often 1–10M trainable; full fine-tune BERT-base ~110M. |
| **Named-entity recognition (NER)** | **~10M–110M** | Fine-tuned BERT-base; or smaller encoder + CRF in the 10–50M range. |
| **Summarization** | **~300M–11B+** | BART, T5-base ~220–300M; larger T5/Pegasus 3B–11B+. |

So: **fill-in-the-blank** is often done with 100M+ params when pre-training; **this project** is a minimal (1–3M) latent-only version. **Conversation** models are usually much larger (100M–B+). **Classification** can be small (1–10M trainable) if using a frozen backbone, or ~110M if fine-tuning BERT-base.

**Why 3M here, when good models in this field are 60M+?** Good fill-in-the-blank / masked-LM models (e.g. DistilBERT ~66M, BERT-base ~110M) are indeed 60M+ and trained on huge corpora. **3M is not “enough” to match that quality.** The doc recommends ~1–3M for *this* setup because: (1) **limited data** — with 3k–20k phrases you have far less signal than BERT’s training data; a 60M+ model would overparameterize and train poorly on that scale. (2) **Minimal proof-of-concept** — this codebase is a JEPA-style latent predictor (encoder + small predictor, no full MLM head), not BERT; it’s aimed at small, local runs. (3) **Stable training** — matching model size to data size (params on the order of effective examples) tends to give stable training and better generalization on that data. **If you want 60M+ quality**, you need much more data (e.g. hundreds of thousands to millions of sentences), longer training, and a larger model (and likely a BERT-like architecture and full MLM objective).

---

## 8. Practical summary

| Question | Short answer |
|----------|----------------|
| How does training work? | Sample phrase → mask one token → encoder gives vector at mask → predictor maps to embedding space → loss = align with true token (MSE + ranking + auxiliary). |
| Is 3k phrases too small? | For a **large** model, yes; for a **~1M param** model, it’s workable. |
| Why is accuracy still random after 50k steps? | Likely model too big for 3k phrases (e.g. 20M params vs ~24k effective examples), so poor generalization or hard optimization. |
| How big should the model be? | For 3k phrases, aim for **~0.5–1M parameters** (e.g. dim 128, 2 layers). Scale up when you have more data. |
| candle-datasets? | Yes: use it for more/better data (e.g. nlp/tinystories or Hub); adapt your loader to feed the same training loop. |

Suggested next steps: **train with a smaller config** (dim 128, 2 layers) on your 3k phrases and see if acc_neg and cosine improve; if they do, then consider adding candle-datasets to scale up data and then model size.

**Example small-config training command** (for ~3k phrases):

```bash
cargo run --release -- --latent data/casual_pairs.txt 50000 16 128 48 24 2 4 4000
```

- `50000` steps, batch `16`, dim `128`, max_ctx `48`, max_tgt `24`, `2` layers, `4` heads, max_vocab `4000`.
- Then run inference with the same dim/layers/heads and seq_len = 48+24 = 72:

```bash
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 128 72 2 4
```
