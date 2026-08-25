# Theoretical diagnosis

The strongest explanation is a conjunction of three structural mismatches:

1. The unrestricted action-conditioned transition is not identifiable from one action per state.
2. Sparse changes and extensive padding make copying a low-risk attractor, while Full V4 never applies categorical pixel supervision to the predicted next state.
3. The live greedy policy is not trained to rank goal-directed actions at all.

Two proposed explanations require qualification:

- Huber’s regression/classification mismatch is real under conditional multimodality or representation aliasing, but the exact synthetic simulator is deterministic at a fully observed `(s,a)`.
- A locality lower bound is valid, but the current architecture already has a nominal receptive field exceeding 64×64. Adaptive recurrence may improve algorithmic computation, but a simple receptive-field deficit is not established.

The repository supports these conclusions. Full V4 excludes factual branch learning, predicted-latent grounding, and ranking losses ([training contract](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-full-v4-training-contract.md:40), [excluded objectives](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-full-v4-training-contract.md:55)). The earlier ADR already records that sequential experience supplied one action per state and that weak action usage followed ([world-core-v2 ADR](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-world-core-v2.md:7)).

One repository discrepancy matters formally: `ArcAction` admits IDs 1–7, with ACTION7 as undo, not merely six atoms ([data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:106)). Moreover, every ACTION6 coordinate is a distinct action atom for identifiability purposes. All theorems below apply after restricting \(A\) to whatever candidate set evaluation actually exposes.

## 1. Identifiability

Let

\[
S=\{0,\ldots,15\}^{64\times64},\qquad
A=\text{legal action atoms},
\]

where an ACTION6 atom includes its coordinate. Let the deterministic environment be

\[
f:S\times A\to S,
\]

and let \(\mu\) be the training distribution over \(S\times A\). A sample is \((s,a,f(s,a))\).

Define the observational equivalence class

\[
[f]_\mu =
\left\{
g:S\times A\to S:
g(s,a)=f(s,a)\text{ for all }(s,a)\in\operatorname{supp}\mu
\right\}.
\]

### Theorem 1: unrestricted non-identifiability

Let \(D\subseteq S\times A\) be the domain on which correctness is required. Assume \(|S|\ge2\). Then \(f|_D\) is identifiable among all functions \(S\times A\to S\) if and only if

\[
D\subseteq\operatorname{supp}\mu.
\]

Equivalently, if some \(x_0=(s_0,a_0)\in D\) has \(\mu(x_0)=0\), there exist \(f,f'\) producing exactly the same observable training distribution while \(f(x_0)\ne f'(x_0)\).

**Proof.** Choose \(y_0\ne f(x_0)\) and define

\[
f'(x)=
\begin{cases}
y_0,&x=x_0,\\
f(x),&x\ne x_0.
\end{cases}
\]

Because \(x_0\) has zero probability, the laws of \((s,a,f(s,a))\) and \((s,a,f'(s,a))\) are identical. Conversely, positive mass at every \(x\in D\) forces every observationally equivalent hypothesis to agree with \(f\) at every such \(x\). ∎

For finite \(D\), the number of consistent restrictions is exactly

\[
|[f]_{\mu,D}|=|S|^{|D\setminus\operatorname{supp}\mu|}.
\]

If the data follow one expert policy \(\pi\), so that the support is contained in the graph

\[
G_\pi=\{(s,\pi(s)):s\in R\},
\]

then, even if every reachable state \(R\) appears, at least

\[
|S|^{|R|(|A|-1)}
\]

different transition tables remain observationally equivalent when each state supplies exactly one action.

The data generators do exactly this along random walks and expert plans: one action is selected and the state is advanced before the next row ([random one-step](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:809), [plan fragments](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1148), [exploration](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1179)). Full V4’s five lessons do not include `factual_branches`, and branch learning is disabled ([train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:44), [recipe](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:588)).

### What condition restores identifiability?

For the unrestricted hypothesis class, the necessary and sufficient condition is full interventional positivity:

\[
\forall(s,a)\in D,\qquad \mu(s,a)>0.
\]

“Two actions per state” is sufficient only if the desired action set at that state has size two. With more actions—or thousands of ACTION6 coordinates—it identifies only the observed contrasts.

For a restricted hypothesis class \(\mathcal H\), the exact condition is weaker and more general:

\[
\operatorname{Res}_{\operatorname{supp}\mu}:\mathcal H\to
S^{\operatorname{supp}\mu}
\quad\text{is injective on }D.
\]

For example, suppose a known feature map \(\phi:S\times A\to X\) is given and every hypothesis factors as \(h=H\circ\phi\). Then it suffices that every feature class \(\phi(s,a)\) occurring in \(D\) is represented in training. This is the precise form of “distributional overlap.” It cannot be assumed for an unrestricted neural network over raw frames.

The repository already contains the right data abstraction: a `BranchGroup` requires byte-identical current frames and distinct actions ([data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:535)); factual batches preserve complete four-action groups ([data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:403)). Four branches are useful contrasts but still do not nonparametrically identify every possible action or coordinate.

**Conclusion:** action collapse is not merely an optimization failure. On unobserved counterfactual pairs, it belongs to an exact observational equivalence class.

## 2. Degenerate minimizers and the copy attractor

Let \(I\) be the \(N\)-pixel gameplay grid, \(C\subseteq I\) the changed pixels, and \(p=|C|/N\). Let \(U=I\setminus C\). Define Huber loss

\[
\rho_\delta(r)=
\begin{cases}
\frac12r^2,&|r|\le\delta,\\
\delta(|r|-\frac12\delta),&|r|>\delta.
\end{cases}
\]

For a predictor \(h\), define

\[
R(h)
=(1-p)w_u\overline L_U(h)+pw_c\overline L_C(h),
\]

where the bars denote per-set averages.

Let \(c_i=x_i\) be the copy predictor and \(y\) the target.

### Theorem 2: exact copy comparison

Define the changed-pixel improvement over copy

\[
I_C(h)=
\frac1{|C|}\sum_{i\in C}
\left[\rho_\delta(c_i-y_i)-\rho_\delta(h_i-y_i)\right]
\]

and unchanged-pixel collateral cost

\[
U_U(h)=
\frac1{|U|}\sum_{i\in U}\rho_\delta(h_i-y_i).
\]

Because \(c_i=y_i\) on \(U\),

\[
R(h)-R(c)
=(1-p)w_uU_U(h)-pw_cI_C(h).
\]

Hence

\[
R(h)<R(c)
\iff
pw_cI_C(h)>(1-p)w_uU_U(h).
\]

Therefore copy is a global minimizer over a restricted class \(\mathcal H\) exactly when

\[
pw_cI_C(h)\le(1-p)w_uU_U(h)
\quad\text{for every }h\in\mathcal H.
\]

Over the unrestricted space of pixel predictions, the oracle \(h=y\) has \(U_U=0\) and \(I_C>0\). Thus copy is neither global nor local whenever \(p>0\) and \(w_c>0\). It becomes an optimizer attractor only because the parameterization couples changed and unchanged pixels, the action Jacobian is weak, gradients cancel, or other objectives dominate.

### Gradient threshold

Huber’s derivative is

\[
\psi_\delta(r)=\operatorname{clip}(r,-\delta,\delta).
\]

For a scalar parameter direction, let \(g_c,g_u\) be the average absolute per-pixel gradient signals from changed and unchanged populations. A sufficient and, under aligned gradients, exact dominance condition is

\[
pw_cg_c>(1-p)w_ug_u.
\]

Equivalently,

\[
\boxed{
\frac{w_c}{w_u}>
\frac{1-p}{p}\frac{g_u}{g_c}
}
\]

and under equal per-pixel signal \(g_c=g_u\),

\[
\boxed{\frac{w_c}{w_u}>\frac{1-p}{p}}.
\]

For \(p=0.02\), the threshold is \(49\); for \(p=0.05\), it is \(19\). Equalizing the two population contributions corresponds to

\[
w_c=\frac1{2p},\qquad
w_u=\frac1{2(1-p)}.
\]

To make changed pixels dominate, use a strictly larger ratio.

There is an important local nuance. Exactly at copy, unchanged pixels have \(\psi_\delta(0)=0\). Therefore, if a feasible parameter direction improves changed pixels to first order while perturbing unchanged pixels only to second order, copy is not a local minimum for any positive \(w_c\). At a parameterized copy solution \(\theta_c\),

\[
\nabla_\theta R(\theta_c)
=
pw_c\,
\mathbb E_C\!\left[
J_i(\theta_c)^\top
\psi_\delta(x_i-y_i)
\right].
\]

It is stationary only if this term vanishes through a zero Jacobian, cancellation, saturation, or representation aliasing. Weak action usage makes a near-zero action-conditioned Jacobian especially plausible.

### Application to Full V4

Full V4 uses one unweighted spatial Huber plus one canonical Huber ([train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3067)). The exact decoder grounds only encoded current and encoded target states; it is not applied to `out.y` ([train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3077)). Predicted-latent grounding is explicitly excluded by the ADR ([training contract](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-full-v4-training-contract.md:55)).

The ordinary synthetic worlds are only 7×7 in training and 8×8 in held-out composition ([generator.rs](/home/stepan/Coding/Personal/Tofy/src/generator.rs:255)), then padded into a 64×64 canvas ([data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:83), [rendering](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:695)). After masking the status row, only \(49/4032\approx1.22\%\) of pixels belong to a 7×7 board. The exact-grounding interface receives frames but no content-dimension mask ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:622)). Subject to the uninspected decoder internals, this strongly suggests that padding also dominates representation grounding.

Thus “copy” is not a mathematical global minimizer of the ideal loss, but it is an extremely low-risk, weak-gradient basin in the actual shared model.

## 3. Regression versus classification

Let \(Y\) be a palette-valued random variable conditional on the model’s available input. Multimodality can arise from stochastic environments, partial observability, state aliasing in the encoder, or pooling distinct true states into the same latent.

### Huber Bayes predictor

For scalar palette encoding, the Huber-optimal predictor is

\[
z_H\in\arg\min_z\mathbb E[\rho_\delta(z-Y)].
\]

When unique, it satisfies

\[
\mathbb E[\operatorname{clip}(z_H-Y,-\delta,\delta)]=0.
\]

It is a robust conditional location estimator. In the all-quadratic regime it is the conditional mean.

For vector latents, the same conclusion applies coordinatewise when the loss decomposes: the optimum is a robust central vector that need not correspond to any valid palette state.

### CE Bayes predictor

For categorical probabilities \(q(k)\),

\[
R_{\rm CE}(q)=\mathbb E[-\log q(Y)]
\]

is uniquely minimized by \(q(k)=P(Y=k)\), up to zero-probability categories. Its argmax decoder returns a conditional mode.

### Theorem 3: mode-versus-location mismatch

Let

\[
k^\star\in\arg\max_k P(Y=k)
\]

and let \(d(z_H)\) be the palette value obtained by decoding the Huber optimum. Then

\[
P(Y=k^\star)\ge P(Y=d(z_H)).
\]

The inequality is strict exactly when \(d(z_H)\) is not a mode. If the unique mode has margin

\[
\gamma=P(Y=k^\star)-\max_{k\ne k^\star}P(Y=k)>0,
\]

then every non-modal Huber decode loses at least \(\gamma\) exact-match probability:

\[
P(Y=k^\star)-P(Y=d(z_H))\ge\gamma.
\]

**Example with the repository’s \(\delta=1\).** Let

\[
P(Y=0)=0.40,\quad
P(Y=1)=0.35,\quad
P(Y=2)=0.25.
\]

For \(z\in[0,1]\), the Huber first-order condition is

\[
0.40z+0.35(z-1)-0.25=0,
\]

so \(z_H=0.8\). Convexity makes this the unique minimizer. Nearest-palette decoding returns \(1\), with exact accuracy \(0.35\). CE returns the mode \(0\), with accuracy \(0.40\).

In the squared-loss regime, \(z_H=\mathbb E[Y]\). Whenever the rounded mean differs from the mode, the same strict mismatch follows.

Two qualifications are essential:

- For the exact deterministic synthetic transition conditioned on a fully observed `(s,a)`, \(Y\) is a point mass. Huber and CE have the same semantic Bayes target. CE does not repair missing counterfactual data.
- Independent per-pixel CE maximizes per-pixel accuracy, not whole-state exact accuracy when pixels are correlated across multiple state modes. A structured state distribution would be needed for that stronger guarantee.

The practical reason to prefer CE here is therefore discrete semantic grounding and calibrated modes under aliasing—not an assertion that the underlying simulator is stochastic.

## 4. Expressivity and depth

### Theorem 4: local light cone

Let a layer \(F_\ell\) be \(r_\ell\)-local: its output at position \(u\) depends only on inputs within distance \(r_\ell\) of \(u\). Then a depth-\(d\) composition is local within radius

\[
R_d\le\sum_{\ell=1}^d r_\ell.
\]

In the homogeneous case \(r_\ell=r\),

\[
R_d\le rd.
\]

**Proof.** Induct on depth. If layer \(d+1\) reads only positions within \(r_{d+1}\) of \(u\), each such intermediate position reads only inputs within \(R_d\). The triangle inequality puts all source inputs within \(R_d+r_{d+1}\) of \(u\). ∎

Consequently, if a target transition has two inputs identical inside the \(R_d\)-ball around \(u\) but requiring different outputs at \(u\), no such network computes the transition exactly.

### Actual Tofy receptive field

The encoder is:

- an 8×8 kernel with stride 8;
- two 3×3, stride-1 convolutions;
- a 1×1 projection.

This produces an 8×8 latent grid ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:254), [encoder](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:347)). Using the standard receptive-field recurrence

\[
R' = R+(k-1)J,\qquad J'=Js,
\]

the encoder gives:

\[
\begin{aligned}
R&=8,\ J=8 &&\text{after patch convolution},\\
R&=24       &&\text{after first 3×3},\\
R&=40       &&\text{after second 3×3}.
\end{aligned}
\]

The recurrent block contains two 3×3 convolutions ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:323)). With two inner steps, one outer step has a deepest path through three residual blocks: two \(z\)-updates and one \(y\)-update ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1096)). With two outer steps, the longest dependency chain has six block invocations, hence twelve additional 3×3 convolutions.

The nominal input-pixel receptive-field side length is therefore

\[
R_{\rm final}
=8+(2+12)(3-1)\cdot8
=232.
\]

Equivalently, the two encoder convolutions give radius two on the 8×8 patch grid; recurrence adds twelve more latent cells of Chebyshev radius. The total nominal latent radius is fourteen, versus an 8×8 grid diameter of seven.

Even one outer step gives

\[
R=8+(2+6)\cdot16=136,
\]

already larger than 64.

Moreover:

- V4 broadcasts the canonical \(B\times C\) readout back into every spatial cell ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:902)).
- ACTION6’s field includes global relative \(x/y\) channels and an active mask, not only a local impulse ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:944)).

Both weaken or violate the pure-locality premise.

### Required recurrent iterations

For a purely local latent architecture with:

- latent diameter \(D_{\rm lat}\),
- encoder radius \(r_{\rm enc}\),
- \(m\) inner steps,
- a two-convolution residual block,

the deepest path adds \(2(m+1)\) latent cells per outer iteration. A sufficient light-cone condition is

\[
o\ge
\left\lceil
\frac{\max(0,D_{\rm lat}-r_{\rm enc})}
     {2(m+1)}
\right\rceil.
\]

Here,

\[
D_{\rm lat}=7,\quad r_{\rm enc}=2,\quad m=2,
\]

so the bound gives

\[
o\ge\lceil5/6\rceil=1.
\]

For a hypothetical pixel-native network with no downsampling and six pixels of propagation per outer iteration, covering distance 63 would instead require eleven outer iterations.

### What remains plausible?

The receptive-field theorem does **not** establish that Full V4 needs more depth. It remains plausible that two outer iterations cannot implement algorithmic operations such as repeated collision propagation, connected-component closure, or stable flood fill despite being able to *see* the whole board. Receptive-field coverage is necessary, not sufficient, for algorithmic computation.

Adaptive recurrence is therefore a principled hypothesis, not a proved fix:

- Use a shared local update, NCA-style.
- Iterate until a residual or discrete-state convergence criterion is met.
- Enforce a hard maximum iteration cap.
- Train across varying sizes and iteration budgets.
- For a DEQ formulation, impose or verify conditions sufficient for stable fixed-point convergence; otherwise existence and uniqueness are not guaranteed.

The code already supports evaluation with extra outer steps without changing weights ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1489)). That matched-depth diagnostic should precede an architectural replacement.

A more immediate distributional issue is that ordinary synthetic training mechanics occur on 7×7 boards, not 64×64 transport tasks. Long-range full-grid ACTION6 examples exist, but they are direct teleports ([data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:842)), not iterative gravity, flood fill, or collision propagation.

## 5. What Epps–Pulley does and does not guarantee

The implementation projects \(B\times D\) or \(T\times B\times D\) canonical latents onto seeded unit directions. For every time slice and projection it compares the empirical characteristic function at finitely many knots with that of \(N(0,1)\) ([sigreg.rs](/home/stepan/Coding/Personal/Tofy/src/p2/sigreg.rs:73), [statistic](/home/stepan/Coding/Personal/Tofy/src/p2/sigreg.rs:119)).

In Full V4:

- there are 1024 directions and 17 knots;
- the analytically zero \(t=0\) knot is skipped;
- current and target canonical states are stacked as two separate time populations;
- predicted states and actions are absent from the EP population.

That population is constructed at [train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3090).

Thus the implementation constrains only:

\[
\widehat\varphi_{v_j}(t_k)
\approx e^{-t_k^2/2}
\]

for the sampled directions \(v_j\) and knots \(t_k\). It does not prove exact 1-D Gaussianity, joint Gaussianity, semantic injectivity, action dependence, transition correctness, or useful dynamics.

### Theorem 5: marginal-regularizer blindness

Let a regularizer have the form

\[
\mathcal R(z)=\Phi(\operatorname{Law}(z(S))),
\]

depending only on the marginal latent population. Suppose \(g:S\to\mathbb R^D\) attains some value \(r\). Define

\[
z'(s,a)=g(s).
\]

Then \(z'\) is action-independent and

\[
\mathcal R(z')=r.
\]

In particular, if \(g\) attains the minimum or zero of the marginal statistic, the action-independent map attains the same value.

**Proof.** The marginal distribution of \(z'(S,A)\) equals that of \(g(S)\). The regularizer receives the same probability law and therefore returns the same value. ∎

This is stronger than a particular counterexample: every perfectly passing marginal population can be duplicated across all actions without affecting EP.

For the ideal population EP statistic, an explicit stochastic counterexample is

\[
z(s,a,\xi)=\xi,\qquad \xi\sim N(0,I_D).
\]

Every projection is exactly standard normal, so ideal EP is zero, while

\[
I(A;Z\mid S)=0
\]

and the latent contains no dynamics information.

There is a finite-model caveat. A deterministic map from finite \(S\) has an atomic distribution and cannot equal a continuous Gaussian exactly. Additionally, V4 RMS-normalizes every canonical sample ([model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:282), [canonical readout](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:651)); exact \(N(0,I)\) has a varying norm, so exact all-direction Gaussianity is incompatible with that fixed-norm support. “Pass perfectly” for the actual implementation can only mean matching its finite directions and knots. The blindness theorem remains exact: whatever minimum finite EP achieves, copying the same state code across actions leaves it unchanged.

A concrete near-counterexample for the actual fixed-norm model is a large spherical codebook \(g(s)\) approximating the uniform sphere. In high dimension, its fixed-direction projections are close to Gaussian while it remains completely action-independent.

### Minimal additional objective

Let

\[
\Delta_\theta(s,a)=
\widehat z_\theta(s,a)-z_\theta(s)
\]

be the predicted latent displacement. On factual same-state branches define

\[
L_{\rm sep}=
\mathbb E_{s,a,b}
\left[
\mathbf 1\{f(s,a)\ne f(s,b)\}
\,[m-\|\Delta_\theta(s,a)-\Delta_\theta(s,b)\|]_+
\right].
\]

If \(m>0\), every covered distinct-effect pair with zero separation loss satisfies

\[
\Delta_\theta(s,a)\ne\Delta_\theta(s,b).
\]

Therefore an action-independent transition cannot attain zero loss on a population containing a positive-mass distinct-effect pair.

This is the minimal *collapse-breaking* addition, but it is not sufficient for correctness. A network could encode arbitrary action IDs. Add an equivalence pull term,

\[
L_{\rm pull}=
\mathbb E\left[
\mathbf 1\{f(s,a)=f(s,b)\}
\|\Delta_\theta(s,a)-\Delta_\theta(s,b)\|^2
\right],
\]

and direct target-displacement supervision. Distinct outcomes are pushed apart; equivalent effects are pulled together. This is essentially the earlier Board Effect design recorded in the V2 ADR ([world-core-v2 ADR](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-world-core-v2.md:20)).

A mutual-information or conditional-variance objective such as

\[
I(A;\Delta_\theta\mid S)>0
\]

rules out total action independence but does not establish outcome-faithful dependence. Interventional branch labels are still needed.

## 6. Greedy one-step sufficiency

Let \(V_g(s')\) be the score used to evaluate a predicted next state for goal \(g\), and let \(\widehat f\) be the model. Define

\[
\widehat Q(s,a)=V_g(\widehat f(s,a)).
\]

A greedy policy chooses

\[
\widehat\pi(s)\in\arg\max_a\widehat Q(s,a).
\]

### Theorem 6: ordinal sufficiency

Let \(\pi^\star(s)\) be the unique correct action at every reachable state. Greedy play chooses it everywhere if and only if

\[
\forall s\in R,\ \forall a\ne\pi^\star(s),\qquad
\widehat Q(s,\pi^\star(s))>\widehat Q(s,a).
\]

Full next-state fidelity is unnecessary. Only the ordering induced by the downstream score matters.

A useful robustness version is the margin theorem. Suppose the true one-step scores are

\[
Q(s,a)=V_g(f(s,a))
\]

with margin

\[
\gamma_s=
Q(s,\pi^\star(s))
-\max_{a\ne\pi^\star(s)}Q(s,a)>0.
\]

If

\[
\sup_a|\widehat Q(s,a)-Q(s,a)|<\gamma_s/2,
\]

then greedy chooses \(\pi^\star(s)\). If \(V_g\) is \(L\)-Lipschitz under a state metric and

\[
d(\widehat f(s,a),f(s,a))\le\varepsilon,
\]

then the sufficient condition becomes

\[
2L\varepsilon<\gamma_s.
\]

This shows why exact pixel prediction is stronger than necessary—but also why small generic prediction error gives no guarantee when action margins are small or the error distorts goal-relevant pixels.

### The actual live policy is not goal-ranked

The repository’s policy explicitly states that it has no trained reward/value head and is not a hidden-goal solver ([arc3_live.rs](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:36)). At evaluation it passes zero goal features for every candidate action ([arc3_live.rs](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:649)) and scores actions as

\[
0.25\,q
+0.30\,\text{reliability}
+0.30(1-\text{noop})
+0.15\,\text{effect},
\]

not by goal progress ([arc3_live.rs](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:663)). Previously tried actions receive an additional heuristic penalty ([arc3_live.rs](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:696)).

Full V4’s \(q\) and reliability targets are exact-decoder transition-correctness labels, not return labels ([train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3167)). The recipe disables PTRM ranking and freezes observer stages away from the world model ([train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:698)).

Therefore the current training objective does not optimize the ordinal condition needed by the evaluator. Even a perfect transition model would not make the current score a correct hidden-goal policy.

### Correct target for a searchless policy

The principled target is an action value on a Markov information state \(b\), where \(b\) may need to include history or a belief over hidden goals:

\[
Q^\star(b,a)
=
\mathbb E\!\left[
r+\gamma\max_{a'}Q^\star(b',a')
\mid b,a
\right].
\]

Train either a listwise objective

\[
L_{\rm list}
=
-\log
\frac{\exp q_\theta(b,a^\star)}
{\sum_a\exp q_\theta(b,a)}
\]

or a pairwise ordinal objective

\[
L_{\rm rank}
=
\sum_{a,b:\,Q^\star(b,a)>Q^\star(b,b)}
[m-(q_\theta(b,a)-q_\theta(b,b))]_+.
\]

If the pairwise loss is zero with positive margin for all reachable action comparisons, greedy argmax is optimal by Theorem 6.

If only one expert action is available, behavioral-cloning CE can rank that demonstrated action on visited states, but it neither identifies counterfactual transitions nor distinguishes multiple equally good actions. Exact simulator values or same-state branch rollouts provide stronger targets.

Training on immediate \(V_g(f(s,a))\) is sufficient only when the environment has a greedy-choice property. It fails when optimal play temporarily moves away from a goal, preserves resources, or chooses an information-gathering action. For hidden goals, \(Q^\star\) must value information in the belief state.

## 7. Lean 4 / mathlib formalizability

The following is mathlib-style pseudocode. Minor API names around `PMF` and finite sums may need adjustment, but the theorem types and proof decompositions are suitable.

### 7.1 Identifiability

```lean
abbrev Grid := (Fin 64 × Fin 64) → Fin 16

def ObsEq {S A : Type*} [DecidableEq S] [DecidableEq A]
    (μ : PMF (S × A)) (f g : S → A → S) : Prop :=
  ∀ x : S × A, μ x ≠ 0 → f x.1 x.2 = g x.1 x.2

theorem nonidentifiable_of_unobserved
    {S A : Type*}
    [Fintype S] [Fintype A] [DecidableEq S] [DecidableEq A]
    [Nontrivial S]
    (μ : PMF (S × A)) (f : S → A → S)
    (s₀ : S) (a₀ : A)
    (hzero : μ (s₀, a₀) = 0) :
    ∃ g : S → A → S,
      ObsEq μ f g ∧ g s₀ a₀ ≠ f s₀ a₀ := by
  classical
  obtain ⟨y₀, hy₀⟩ := exists_ne (f s₀ a₀)
  let g := fun s a =>
    if s = s₀ ∧ a = a₀ then y₀ else f s a
  refine ⟨g, ?_, ?_⟩
  · intro x hx
    -- x cannot equal (s₀,a₀), because hzero contradicts hx.
    sorry
  · simp [g, hy₀]

theorem identifiable_of_full_support
    {S A : Type*}
    [Fintype S] [Fintype A] [DecidableEq S] [DecidableEq A]
    (μ : PMF (S × A)) (f g : S → A → S)
    (R : Finset S)
    (hpos : ∀ s ∈ R, ∀ a, μ (s, a) ≠ 0)
    (heq : ObsEq μ f g) :
    ∀ s ∈ R, ∀ a, f s a = g s a := by
  intro s hs a
  exact heq (s, a) (hpos s hs a)
```

Difficulty: elementary finite mathematics. The class-cardinality theorem needs finite function-counting lemmas but no probability theory beyond support.

Proof strategy: change one unobserved table entry; restoration is direct application of positive support.

### 7.2 Copy comparison and weighting threshold

```lean
def huber (δ r : ℝ) : ℝ :=
  if |r| ≤ δ then r^2 / 2 else δ * (|r| - δ / 2)

def pixelRisk {I : Type*} [Fintype I] [DecidableEq I]
    (C : Finset I) (wu wc : ℝ)
    (x y h : I → ℝ) : ℝ :=
  wu * ∑ i in Cᶜ, huber 1 (h i - y i) +
  wc * ∑ i in C,  huber 1 (h i - y i)

def changedImprovement {I : Type*} [Fintype I] [DecidableEq I]
    (C : Finset I) (x y h : I → ℝ) : ℝ :=
  ∑ i in C, (huber 1 (x i - y i) - huber 1 (h i - y i))

def unchangedCollateral {I : Type*} [Fintype I] [DecidableEq I]
    (C : Finset I) (y h : I → ℝ) : ℝ :=
  ∑ i in Cᶜ, huber 1 (h i - y i)

theorem competitor_beats_copy_iff
    {I : Type*} [Fintype I] [DecidableEq I]
    (C : Finset I) (wu wc : ℝ)
    (x y h : I → ℝ)
    (hunchanged : ∀ i ∈ Cᶜ, x i = y i) :
    pixelRisk C wu wc x y h < pixelRisk C wu wc x y x ↔
      wc * changedImprovement C x y h >
        wu * unchangedCollateral C y h := by
  unfold pixelRisk changedImprovement unchangedCollateral
  -- Split finite sums and use huber 1 0 = 0.
  sorry

theorem changed_signal_dominates
    (p wu wc gu gc : ℝ)
    (hp0 : 0 < p) (hp1 : p < 1)
    (hwu : 0 < wu) (hgc : 0 < gc) :
    p * wc * gc > (1 - p) * wu * gu ↔
      wc / wu > ((1 - p) / p) * (gu / gc) := by
  field_simp
  nlinarith
```

Difficulty: comparison identity and threshold are elementary. A general local-minimum theorem for neural parameterizations requires multivariable derivatives and is moderately difficult.

Proof strategy: split changed/unchanged finite sums; for the gradient result, define Huber’s derivative and apply chain rule.

### 7.3 Huber versus CE

```lean
def huberRisk {α : Type*} [Fintype α]
    (P : PMF α) (embed : α → ℝ) (δ z : ℝ) : ℝ :=
  ∑ a, (P a).toReal * huber δ (z - embed a)

def exactAccuracy {α : Type*} (P : PMF α) (a : α) : ℝ :=
  (P a).toReal

def crossEntropy {α : Type*} [Fintype α]
    (P : PMF α) (q : α → ℝ) : ℝ :=
  -∑ a, (P a).toReal * Real.log (q a)

theorem mode_maximizes_exact_accuracy
    {α : Type*} [Fintype α]
    (P : PMF α) (mode decoded : α)
    (hmode : ∀ a, P a ≤ P mode) :
    exactAccuracy P decoded ≤ exactAccuracy P mode := by
  exact ENNReal.toReal_mono (PMF.apply_ne_top P decoded) (hmode decoded)

theorem strict_mode_gap
    {α : Type*} [Fintype α]
    (P : PMF α) (mode decoded : α)
    (hstrict : P decoded < P mode) :
    exactAccuracy P decoded < exactAccuracy P mode := by
  exact ENNReal.toReal_strict_mono
    (PMF.apply_ne_top P decoded) (PMF.apply_ne_top P mode) hstrict

theorem categorical_ce_bayes
    {α : Type*} [Fintype α] [DecidableEq α]
    (P : PMF α) :
    ∀ q : α → ℝ,
      (∀ a, 0 < q a) →
      (∑ a, q a = 1) →
      crossEntropy P (fun a => (P a).toReal) ≤ crossEntropy P q := by
  -- Rewrite the excess risk as KL(P || q) and use Gibbs' inequality.
  sorry

theorem huber_counterexample :
    let P : Fin 3 → ℝ := fun
      | 0 => 2 / 5
      | 1 => 7 / 20
      | 2 => 1 / 4
    let R := fun z =>
      ∑ k : Fin 3, P k * huber 1 (z - k.val)
    IsLeast {z : ℝ | True} (4 / 5) ∧
      P 1 < P 0 := by
  -- Prove convexity; show the clipped derivative changes sign at 4/5.
  sorry
```

Difficulty: exact-match claims and the three-point Huber example are elementary real analysis. CE optimality needs logarithms, KL divergence, and mathlib’s finite probability machinery.

Proof strategy: Gibbs inequality for CE; piecewise clipped derivative plus convexity for Huber.

### 7.4 Locality

```lean
def LocalAt
    {V X : Type*} [PseudoMetricSpace V]
    (r : ℝ) (F : (V → X) → V → X) : Prop :=
  ∀ x y u,
    (∀ v, dist v u ≤ r → x v = y v) →
    F x u = F y u

theorem local_comp
    {V X : Type*} [PseudoMetricSpace V]
    (F G : (V → X) → V → X)
    (rF rG : ℝ)
    (hF : LocalAt rF F)
    (hG : LocalAt rG G) :
    LocalAt (rF + rG) (fun x => F (G x)) := by
  intro x y u hxy
  apply hF
  intro v hv
  apply hG
  intro w hw
  apply hxy
  exact le_trans (dist_triangle w v u) (add_le_add hw hv)

theorem repeated_local_lightcone
    {V X : Type*} [PseudoMetricSpace V]
    (F : (V → X) → V → X) (r : ℝ)
    (hF : LocalAt r F) :
    ∀ d : ℕ, LocalAt (d * r) (Function.iterate F d) := by
  intro d
  induction d with
  | zero => simp [LocalAt]
  | succ d ih =>
      simpa [Function.iterate_succ_apply, Nat.cast_succ,
        add_mul] using local_comp F (Function.iterate F d) r (d * r) hF ih

theorem current_receptive_field_arithmetic :
    8 + (2 + 12) * (3 - 1) * 8 = 232 := by
  norm_num
```

Difficulty: the light-cone theorem is elementary once locality is defined. Formalizing exact padded convolution semantics and linking it to Candle code would be substantially harder.

Proof strategy: induction and the triangle inequality; receptive-field arithmetic is normalization.

### 7.5 EP blindness and separation

```lean
def marginalReg {Z : Type*}
    (R : PMF Z → ℝ) (P : PMF Z) : ℝ :=
  R P

theorem action_independent_has_state_marginal
    {S A Z : Type*}
    (μ : PMF (S × A)) (g : S → Z) :
    μ.map (fun sa => g sa.1) =
      (μ.map Prod.fst).map g := by
  simpa using PMF.map_map μ Prod.fst g

theorem marginal_regularizer_blind
    {S A Z : Type*}
    (R : PMF Z → ℝ) (μ : PMF (S × A)) (g : S → Z) :
    R (μ.map (fun sa => g sa.1)) =
      R ((μ.map Prod.fst).map g) := by
  rw [action_independent_has_state_marginal]

def sepLossTerm {Z : Type*} [PseudoMetricSpace Z]
    (m : ℝ) (za zb : Z) : ℝ :=
  max 0 (m - dist za zb)

theorem zero_separation_forces_difference
    {Z : Type*} [PseudoMetricSpace Z]
    (m : ℝ) (hm : 0 < m) (za zb : Z)
    (hzero : sepLossTerm m za zb = 0) :
    za ≠ zb := by
  intro heq
  subst heq
  simp [sepLossTerm, hm] at hzero
```

Difficulty: marginal blindness and the margin theorem are elementary. Formalizing ideal EP, characteristic functions, Gaussian measures, and the fixed-norm incompatibility needs mathlib measure theory and is hard.

Proof strategy: functoriality of `PMF.map`; a zero positive-margin hinge contradicts zero distance.

### 7.6 Greedy ordinal correctness

```lean
def RanksFirst {S A : Type*}
    (q : S → A → ℝ) (π : S → A) (s : S) : Prop :=
  ∀ a, a ≠ π s → q s (π s) > q s a

theorem every_maximizer_is_correct
    {S A : Type*} [Fintype A]
    (q : S → A → ℝ) (π : S → A) (s : S)
    (hrank : RanksFirst q π s)
    (ahat : A)
    (hmax : ∀ a, q s a ≤ q s ahat) :
    ahat = π s := by
  by_contra hne
  have h₁ := hrank ahat hne
  have h₂ := hmax (π s)
  linarith

theorem ranking_survives_uniform_error
    {A : Type*} [Fintype A]
    (Q Qhat : A → ℝ) (astar : A) (γ ε : ℝ)
    (hmargin : ∀ a, a ≠ astar → Q astar ≥ Q a + γ)
    (herr : ∀ a, |Qhat a - Q a| ≤ ε)
    (hgap : 2 * ε < γ) :
    ∀ a, a ≠ astar → Qhat astar > Qhat a := by
  intro a hne
  have hm := hmargin a hne
  have he₁ := herr astar
  have he₂ := herr a
  rw [abs_le] at he₁ he₂
  linarith
```

Difficulty: elementary. Extending it to repeated play requires an induction over the reachable transition system. Belief-state optimality or POMDP results are considerably harder.

Proof strategy: strict-order contradiction; the error theorem is a triangle-inequality calculation.

## 8. Prioritized interventions

### 1. Require interventional action coverage

Reintroduce complete same-state Branch Groups, or another data design satisfying an explicit restricted-class overlap condition.

This is the highest-priority intervention because Theorem 1 establishes impossibility without it. No representation regularizer, optimizer, or architectural action field can recover arbitrary unobserved counterfactuals.

For ACTION6, four coordinate branches do not identify all coordinates nonparametrically. Use structured coordinate coverage and preregister the equivariance/generalization assumptions being relied upon.

The earlier V2/V3 treatments were empirically rejected by frozen gates ([ADR 0002](/home/stepan/Coding/Personal/Tofy/docs/adr/0002-resolved-experiments-and-factual-batches.md:5)). That is evidence against those complete treatment bundles, not a refutation of the identifiability theorem. A new experiment should isolate the data intervention from rejected architectural and loss changes.

### 2. Apply categorical supervision to the predicted next state, with content/change balancing

Decode `out.y` and apply masked per-pixel palette CE to the actual next frame. Do not rely only on current/target encoder grounding.

Use

\[
L_{\rm pred}
=
(1-p)w_u\,\overline{\rm CE}_U+
pw_c\,\overline{\rm CE}_C
\]

with

\[
w_c/w_u\ge(1-p)/p
\]

for equalized signal, and larger for changed-pixel dominance. Estimate \(p\) on gameplay content, not the padded 64×64 canvas. Mask status and proven padding separately.

Theorem 2 justifies the weighting; Theorem 3 justifies categorical semantics under aliasing. This is the intervention most directly targeted at the reported 2–5% changed-pixel plateau.

### 3. Add paired Board-Effect displacement separation

On branch pairs, use:

- push margins for distinct board effects;
- pull losses for equivalent effects;
- direct prediction loss for each factual branch;
- optionally action/coordinate recovery from displacement as a diagnostic auxiliary.

Theorem 5 proves that a positive-margin separation loss rules out action-independent predictions on covered distinct-effect pairs. Pulling equivalent outcomes prevents an arbitrary action-ID code from satisfying the push term.

This intervention depends on priority 1; without same-state effect labels it cannot distinguish faithful action use from decorative action encoding.

### 4. Train the policy on action-value ordering

Replace or supplement the current reliability/effect heuristic with a goal- or belief-conditioned \(Q\) head trained using listwise or pairwise action rankings.

For exact simulator lessons, compute action values or at least branch-relative advantages. For hidden-goal tasks, condition on history/belief and include information value. Theorem 6 gives the exact guarantee: correct ordinal ranking is sufficient even when predicted states are imperfect.

This may have less impact on changed-pixel exact accuracy than priorities 1–3, but it is a necessary intervention for searchless greedy play. The current live score cannot become a guaranteed goal solver merely by improving transition fidelity.

### 5. Test adaptive recurrence only after a premise check

Run the existing matched-extra-outer-step diagnostic on specifically constructed long-range tasks. Check whether:

- extra iterations improve frozen-checkpoint predictions;
- outputs converge or oscillate;
- performance depends on transport distance;
- action sensitivity grows with depth.

If those checks are positive, introduce shared adaptive recurrence with an explicit stopping rule and maximum depth. Theorem 4 justifies adaptive depth for genuinely local architectures, but the current receptive-field calculation does not establish a light-cone bottleneck. Consequently this is a lower-confidence intervention than data, loss, and ranking alignment.

## Final assessment

The following are proved from elementary finite mathematics:

- unrestricted transition non-identifiability without pair coverage;
- the exact copy-versus-competitor inequality and reweighting threshold;
- CE mode optimality for exact categorical accuracy;
- the locality/light-cone bound;
- marginal EP’s inability to imply action dependence;
- ordinal sufficiency for greedy action selection.

The following are supported interpretations, not theorems about the trained checkpoint:

- sparse changes and padding caused the observed optimization plateau;
- weak action gradients trapped training near copy;
- categorical predicted-state loss will materially improve ARC transfer;
- adaptive recurrence will help algorithmic dynamics.

The simple claim that “two outer steps cannot propagate across 64 pixels” is disproved for this architecture’s nominal receptive field. The stronger causal diagnosis is: the system lacks interventional identifiability, directly supervises predicted states only in continuous latent space with sparse semantic change, regularizes only marginal state geometry, and evaluates with an objective that is not goal-value ranking.
