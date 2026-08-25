import Mathlib

/-!
# Theorem B: cross-entropy upper-bounds exact-match error

For a probability vector `p` over a finite palette with true class `y` and
argmax prediction `ŷ`, the 0/1 exact-match error is bounded by the
cross-entropy in bits: `[ŷ ≠ y] ≤ (-log (p y)) / log 2`.  Key step: if the
argmax misses `y` then `p y ≤ 1/2`.  Summing over a finite set of pixels,
the changed-pixel cross-entropy controls exact-transition error.
-/

namespace TofyFormal

open Finset

variable {Color : Type*} [Fintype Color] [DecidableEq Color]

/-- If the argmax `ŷ` of a probability vector differs from `y`, then the
mass at `y` is at most `1/2` (since `p y ≤ p ŷ` and `p ŷ + p y ≤ 1`). -/
lemma prob_true_le_half_of_argmax_ne
    {p : Color → ℝ} (hnn : ∀ c, 0 ≤ p c) (hsum : ∑ c, p c = 1)
    {y yhat : Color} (hmax : ∀ c, p c ≤ p yhat) (hne : yhat ≠ y) :
    p y ≤ 1 / 2 := by
  have hpair : p yhat + p y ≤ 1 := by
    have hsub : ∑ c ∈ ({yhat, y} : Finset Color), p c ≤ ∑ c, p c :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        fun c _ _ => hnn c
    rwa [Finset.sum_pair hne, hsum] at hsub
  linarith [hmax y]

/-- If the argmax misses the true class, the cross-entropy is at least one
bit: `log 2 ≤ -log (p y)`. -/
lemma log_two_le_neg_log_of_argmax_ne
    {p : Color → ℝ} (hnn : ∀ c, 0 ≤ p c) (hsum : ∑ c, p c = 1)
    {y yhat : Color} (hpos : 0 < p y) (hmax : ∀ c, p c ≤ p yhat)
    (hne : yhat ≠ y) : Real.log 2 ≤ -Real.log (p y) := by
  have hhalf : p y ≤ 1 / 2 := prob_true_le_half_of_argmax_ne hnn hsum hmax hne
  have hmul : Real.log 2 + Real.log (p y) = Real.log (2 * p y) :=
    (Real.log_mul two_ne_zero hpos.ne').symm
  have hle : Real.log (2 * p y) ≤ 0 :=
    Real.log_nonpos (by positivity) (by linarith)
  linarith

omit [DecidableEq Color] in
/-- The true-class probability is at most one, so `-log (p y)` is
nonnegative. -/
lemma neg_log_nonneg {p : Color → ℝ} (hnn : ∀ c, 0 ≤ p c)
    (hsum : ∑ c, p c = 1) (y : Color) : 0 ≤ -Real.log (p y) := by
  have h1 : p y ≤ 1 := hsum ▸ Finset.single_le_sum (fun c _ => hnn c) (mem_univ y)
  have := Real.log_nonpos (hnn y) h1
  linarith

/-- **Theorem B (pointwise).** The 0/1 exact-match error of the argmax
prediction is bounded by the cross-entropy measured in bits. -/
theorem indicator_ne_le_ce
    {p : Color → ℝ} (hnn : ∀ c, 0 ≤ p c) (hsum : ∑ c, p c = 1)
    {y yhat : Color} (hpos : 0 < p y) (hmax : ∀ c, p c ≤ p yhat) :
    (if yhat ≠ y then (1 : ℝ) else 0) ≤ -Real.log (p y) / Real.log 2 := by
  have h2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  by_cases hne : yhat = y
  · simp only [hne, ne_eq, not_true_eq_false, if_false]
    exact div_nonneg (neg_log_nonneg hnn hsum y) h2.le
  · rw [if_pos hne, le_div_iff₀ h2, one_mul]
    exact log_two_le_neg_log_of_argmax_ne hnn hsum hpos hmax hne

/-- **Theorem B (summed).** Over a finite pixel set `M`, the 0/1 indicator
that *some* pixel's argmax prediction is wrong is bounded by the summed
cross-entropy in bits: changed-pixel CE controls exact-transition error. -/
theorem indicator_exists_ne_le_sum_ce
    {ι : Type*} (M : Finset ι) (p : ι → Color → ℝ) (y yhat : ι → Color)
    (hnn : ∀ i ∈ M, ∀ c, 0 ≤ p i c) (hsum : ∀ i ∈ M, ∑ c, p i c = 1)
    (hpos : ∀ i ∈ M, 0 < p i (y i))
    (hmax : ∀ i ∈ M, ∀ c, p i c ≤ p i (yhat i)) :
    (if ∃ i ∈ M, yhat i ≠ y i then (1 : ℝ) else 0)
      ≤ (∑ i ∈ M, -Real.log (p i (y i))) / Real.log 2 := by
  have h2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hterm : ∀ i ∈ M, 0 ≤ -Real.log (p i (y i)) :=
    fun i hi => neg_log_nonneg (hnn i hi) (hsum i hi) (y i)
  by_cases hex : ∃ i ∈ M, yhat i ≠ y i
  · obtain ⟨i₀, hi₀, hne₀⟩ := hex
    rw [if_pos ⟨i₀, hi₀, hne₀⟩, le_div_iff₀ h2, one_mul]
    calc Real.log 2
        ≤ -Real.log (p i₀ (y i₀)) :=
          log_two_le_neg_log_of_argmax_ne (hnn i₀ hi₀) (hsum i₀ hi₀)
            (hpos i₀ hi₀) (hmax i₀ hi₀) hne₀
      _ ≤ ∑ i ∈ M, -Real.log (p i (y i)) :=
          Finset.single_le_sum hterm hi₀
  · rw [if_neg hex]
    exact div_nonneg (Finset.sum_nonneg hterm) h2.le

end TofyFormal
