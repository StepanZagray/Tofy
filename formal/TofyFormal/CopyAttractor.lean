import Mathlib

/-!
# Theorem D: copy-map attractor and the reweighting threshold

Pixels split into a changed set `M` and an unchanged set `U`.  The copy
predictor is perfect on `U` and pays at least `c > 0` on each changed
pixel.  Under weights `w_c` (changed) / `w_u` (unchanged) its total loss is
at least `w_c * |M| * c`, while a correct predictor pays `0`; but under
*uniform* weighting its **average** loss is at most any `ε > 0` once
`|U| ≥ w * |M| * L_max / ε` — the copy map is `ε`-optimal when changes are
rare, which is precisely the attractor that changed-pixel reweighting
removes.
-/

namespace TofyFormal

open Finset

variable {ι : Type*} [DecidableEq ι]

/-- **Theorem D (lower bound).** With weight `w_c ≥ 0` on changed pixels,
the copy predictor's total weighted loss is at least `w_c * |M| * c`. -/
theorem copy_weighted_loss_lower
    (M U : Finset ι) (hMU : Disjoint M U) (ℓ : ι → ℝ) {c : ℝ}
    {w_c : ℝ} (w_u : ℝ) (hwc : 0 ≤ w_c)
    (hU : ∀ i ∈ U, ℓ i = 0) (hM : ∀ i ∈ M, c ≤ ℓ i) :
    w_c * M.card * c ≤ ∑ i ∈ M ∪ U, (if i ∈ M then w_c else w_u) * ℓ i := by
  rw [Finset.sum_union hMU]
  have hUzero : ∑ i ∈ U, (if i ∈ M then w_c else w_u) * ℓ i = 0 :=
    Finset.sum_eq_zero fun i hi => by rw [hU i hi, mul_zero]
  have hMsum : ∑ i ∈ M, (if i ∈ M then w_c else w_u) * ℓ i
      = w_c * ∑ i ∈ M, ℓ i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i hi => by rw [if_pos hi]
  have hcard : (M.card : ℝ) * c ≤ ∑ i ∈ M, ℓ i := by
    have := Finset.card_nsmul_le_sum M ℓ c hM
    rwa [nsmul_eq_mul] at this
  rw [hMsum, hUzero, add_zero, mul_assoc]
  exact mul_le_mul_of_nonneg_left hcard hwc

omit [DecidableEq ι] in
/-- A predictor with zero per-pixel loss has zero weighted total loss, for
any weights. -/
theorem correct_predictor_zero_loss
    (s : Finset ι) (w ℓ : ι → ℝ) (h : ∀ i ∈ s, ℓ i = 0) :
    ∑ i ∈ s, w i * ℓ i = 0 :=
  Finset.sum_eq_zero fun i hi => by rw [h i hi, mul_zero]

/-- **Theorem D (ε-optimality under uniform weighting).** If per-pixel copy
loss is `0` on `U` and at most `L_max` on `M`, then for any `ε > 0`, as soon
as `|U| ≥ w * |M| * L_max / ε` the uniformly-weighted *average* copy loss is
at most `ε`.  As `|U| / |M| → ∞` the copy map becomes ε-optimal for every
`ε`: the copy attractor. -/
theorem copy_average_loss_le_of_rare_changes
    (M U : Finset ι) (hMU : Disjoint M U) (ℓ : ι → ℝ)
    {w Lmax ε : ℝ} (hw : 0 ≤ w) (hε : 0 < ε)
    (hM : ∀ i ∈ M, ℓ i ≤ Lmax) (hU : ∀ i ∈ U, ℓ i = 0)
    (hbig : w * M.card * Lmax / ε ≤ (U.card : ℝ)) :
    (∑ i ∈ M ∪ U, w * ℓ i) / ((M.card : ℝ) + U.card) ≤ ε := by
  have htot : ∑ i ∈ M ∪ U, w * ℓ i ≤ w * M.card * Lmax := by
    rw [Finset.sum_union hMU]
    have hUzero : ∑ i ∈ U, w * ℓ i = 0 :=
      Finset.sum_eq_zero fun i hi => by rw [hU i hi, mul_zero]
    have hMle : ∑ i ∈ M, w * ℓ i ≤ M.card • (w * Lmax) :=
      Finset.sum_le_card_nsmul M _ _
        fun i hi => mul_le_mul_of_nonneg_left (hM i hi) hw
    rw [hUzero, add_zero]
    calc ∑ i ∈ M, w * ℓ i ≤ M.card • (w * Lmax) := hMle
      _ = w * M.card * Lmax := by rw [nsmul_eq_mul]; ring
  have hεU : w * M.card * Lmax ≤ ε * U.card := by
    rw [div_le_iff₀ hε] at hbig
    linarith
  rcases Nat.eq_zero_or_pos (M.card + U.card) with h0 | hpos
  · obtain ⟨hM0, hU0⟩ := Nat.add_eq_zero_iff.mp h0
    rw [Finset.card_eq_zero.mp hM0, Finset.card_eq_zero.mp hU0]
    simp [hε.le]
  · have hden : (0 : ℝ) < (M.card : ℝ) + U.card := by
      have h : (0 : ℝ) < ((M.card + U.card : ℕ) : ℝ) := by exact_mod_cast hpos
      push_cast at h
      linarith
    rw [div_le_iff₀ hden]
    have hUle : (U.card : ℝ) ≤ (M.card : ℝ) + U.card := by
      have h : (0 : ℝ) ≤ (M.card : ℝ) := Nat.cast_nonneg _
      linarith
    calc ∑ i ∈ M ∪ U, w * ℓ i
        ≤ w * M.card * Lmax := htot
      _ ≤ ε * U.card := hεU
      _ ≤ ε * ((M.card : ℝ) + U.card) := by nlinarith

end TofyFormal
