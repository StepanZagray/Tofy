import Mathlib

/-!
# Theorem E: group symmetrization does not increase risk (finite version)

A finite group `T` acts on finite inputs `X`; predictions live in a real
vector space `V` (e.g. per-pixel color logits `Cell → Color → ℝ`).  The
group acts on predictions through an arbitrary family `ρ : T → V → V`
(for grid translations: index shifting), tied to the loss by the
equivariance identity `ℓ x (ρ t v) = ℓ (t • x) v` — "evaluating a
transformed prediction at `x` is evaluating the prediction at the
transformed input", which encodes label equivariance.  If the data weights
`μ` are invariant and the loss is convex in the prediction, then the
group-averaged predictor `f̄ x = (1/|T|) ∑ t, ρ t (f (t • x))` has risk at
most that of `f`.  Proof: pointwise Jensen, then unfold via equivariance,
`μ`-invariance, and reindexing by the bijection `x ↦ t • x`.
-/

namespace TofyFormal

open Finset

/-- **Theorem E.** Group-averaging a predictor does not increase the risk
`∑ x, μ x * ℓ x (f x)` when `μ` is action-invariant, the loss is convex in
the prediction, and the loss is equivariant under the output action `ρ`. -/
theorem symmetrized_risk_le
    {T X V : Type*} [Group T] [Fintype T] [Fintype X]
    [MulAction T X] [AddCommGroup V] [Module ℝ V]
    (μ : X → ℝ) (hμ0 : ∀ x, 0 ≤ μ x)
    (hμinv : ∀ (t : T) (x : X), μ (t • x) = μ x)
    (ℓ : X → V → ℝ) (hconv : ∀ x, ConvexOn ℝ Set.univ (ℓ x))
    (ρ : T → V → V)
    (hequiv : ∀ (t : T) (x : X) (v : V), ℓ x (ρ t v) = ℓ (t • x) v)
    (f : X → V) :
    ∑ x, μ x * ℓ x (∑ t : T, (Fintype.card T : ℝ)⁻¹ • ρ t (f (t • x)))
      ≤ ∑ x, μ x * ℓ x (f x) := by
  have hcard : (0 : ℝ) < (Fintype.card T : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hw0 : (0 : ℝ) ≤ (Fintype.card T : ℝ)⁻¹ := by positivity
  have hwsum : ∑ _t : T, (Fintype.card T : ℝ)⁻¹ = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_inv_cancel₀ hcard.ne']
  -- Pointwise Jensen: the loss of the average is at most the average loss.
  have hjensen : ∀ x : X,
      ℓ x (∑ t : T, (Fintype.card T : ℝ)⁻¹ • ρ t (f (t • x)))
        ≤ ∑ t : T, (Fintype.card T : ℝ)⁻¹ * ℓ x (ρ t (f (t • x))) := by
    intro x
    have h := (hconv x).map_sum_le (t := Finset.univ)
      (w := fun _ : T => (Fintype.card T : ℝ)⁻¹)
      (p := fun t : T => ρ t (f (t • x)))
      (fun t _ => hw0) hwsum (fun t _ => Set.mem_univ _)
    simpa [smul_eq_mul] using h
  -- Reindexing along the bijection `x ↦ t • x`.
  have hreindex : ∀ t : T,
      ∑ x, μ (t • x) * ℓ (t • x) (f (t • x)) = ∑ x, μ x * ℓ x (f x) :=
    fun t => Fintype.sum_bijective (fun x => t • x) (MulAction.bijective t)
      (fun x => μ (t • x) * ℓ (t • x) (f (t • x)))
      (fun x => μ x * ℓ x (f x)) (fun _ => rfl)
  calc ∑ x, μ x * ℓ x (∑ t : T, (Fintype.card T : ℝ)⁻¹ • ρ t (f (t • x)))
      ≤ ∑ x, μ x * ∑ t : T, (Fintype.card T : ℝ)⁻¹ * ℓ x (ρ t (f (t • x))) :=
        Finset.sum_le_sum fun x _ =>
          mul_le_mul_of_nonneg_left (hjensen x) (hμ0 x)
    _ = ∑ t : T, ∑ x, (Fintype.card T : ℝ)⁻¹ *
          (μ x * ℓ x (ρ t (f (t • x)))) := by
        rw [← Finset.sum_comm]
        exact Finset.sum_congr rfl fun x _ => by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun t _ => by ring
    _ = ∑ t : T, (Fintype.card T : ℝ)⁻¹ * ∑ x, μ x * ℓ x (f x) := by
        refine Finset.sum_congr rfl fun t _ => ?_
        rw [← Finset.mul_sum]
        congr 1
        calc ∑ x, μ x * ℓ x (ρ t (f (t • x)))
            = ∑ x, μ (t • x) * ℓ (t • x) (f (t • x)) :=
              Finset.sum_congr rfl fun x _ => by rw [hequiv, hμinv]
          _ = ∑ x, μ x * ℓ x (f x) := hreindex t
    _ = ∑ x, μ x * ℓ x (f x) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← mul_assoc,
          mul_inv_cancel₀ hcard.ne', one_mul]

end TofyFormal
