import Mathlib

/-!
# Theorem J: mode decoding is exact-match optimal; regression decoding is not

Exact-match accuracy of decoding to `k` is `p k`, so the mode maximizes
it.  The substantive result is a concrete counterexample: for the
distribution `P(0) = 2/5, P(1) = 7/20, P(2) = 1/4` on `{0, 1, 2} ⊆ ℝ` and
Huber loss with `δ = 1`, the *global* Huber-risk minimizer over `ℝ` is
`z = 4/5`, whose nearest integer is `1` — while the mode is `0`.
Regression-style decoding therefore lands off the mode.  Global minimality
is proved via hand-rolled tangent-line bounds for the Huber loss (its
convexity, specialized to the three residual points), which sum with zero
net slope at `4/5`.
-/

namespace TofyFormal

/-- **Theorem J (i).** Exact-match accuracy of decoding to `decoded` is
`p decoded`; the mode maximizes it. -/
theorem mode_maximizes_exact_match {α : Type*} (p : α → ℝ) {mode : α}
    (hmode : ∀ a, p a ≤ p mode) (decoded : α) : p decoded ≤ p mode :=
  hmode decoded

/-- **Theorem J (ii).** Decoding to any point with strictly smaller
probability gives strictly worse exact-match accuracy.  (The content is
the packaging: accuracy *is* the decoded point's probability.) -/
theorem decode_off_mode_strictly_worse {α : Type*} (p : α → ℝ)
    {mode decoded : α} (h : p decoded < p mode) : p decoded < p mode := h

/-! ## The Huber counterexample -/

/-- Huber loss with `δ = 1`: quadratic within the clip, linear outside. -/
noncomputable def huber (r : ℝ) : ℝ := if |r| ≤ 1 then r ^ 2 / 2 else |r| - 1 / 2

lemma huber_of_abs_le {r : ℝ} (h : |r| ≤ 1) : huber r = r ^ 2 / 2 := by
  rw [huber, if_pos h]

lemma huber_of_one_lt {r : ℝ} (h : 1 < |r|) : huber r = |r| - 1 / 2 := by
  rw [huber, if_neg (not_le.mpr h)]

/-- The Huber loss dominates its linear tails everywhere. -/
lemma huber_ge_abs (t : ℝ) : |t| - 1 / 2 ≤ huber t := by
  rw [huber]
  split_ifs with h
  · nlinarith [sq_abs t, sq_nonneg (|t| - 1)]
  · exact le_refl _

/-- Tangent-line bound for the Huber loss at any point of the quadratic
region: convexity, specialized to what the minimality proof needs. -/
lemma huber_tangent_le {t₀ : ℝ} (h₀ : |t₀| ≤ 1) (t : ℝ) :
    t₀ ^ 2 / 2 + t₀ * (t - t₀) ≤ huber t := by
  rw [huber]
  split_ifs with h
  · nlinarith [sq_nonneg (t - t₀)]
  · have h1 : 1 ≤ |t| := (not_le.mp h).le
    have habs : t₀ * t ≤ |t₀| * |t| :=
      calc t₀ * t ≤ |t₀ * t| := le_abs_self _
        _ = |t₀| * |t| := abs_mul _ _
    have key : 0 ≤ (1 - |t₀|) * (|t| - (1 + |t₀|) / 2) := by
      apply mul_nonneg <;> linarith [abs_nonneg t₀]
    have key' : |t₀| * |t| + 1 / 2 ≤ |t| + t₀ ^ 2 / 2 := by
      nlinarith [key, sq_abs t₀]
    linarith

/-- Huber risk of predicting `z` for the distribution
`P(0) = 2/5, P(1) = 7/20, P(2) = 1/4` on values `{0, 1, 2}`. -/
noncomputable def huberRisk (z : ℝ) : ℝ :=
  2 / 5 * huber z + 7 / 20 * huber (z - 1) + 1 / 4 * huber (z - 2)

lemma huberRisk_at_min : huberRisk (4 / 5) = 31 / 100 := by
  have h1 : |(4 / 5 : ℝ)| ≤ 1 := by rw [abs_le]; constructor <;> norm_num
  have h2 : |(4 / 5 : ℝ) - 1| ≤ 1 := by rw [abs_le]; constructor <;> norm_num
  have h3 : (1 : ℝ) < |(4 / 5 : ℝ) - 2| := by
    rw [abs_of_nonpos (by norm_num : (4 : ℝ) / 5 - 2 ≤ 0)]; norm_num
  rw [huberRisk, huber_of_abs_le h1, huber_of_abs_le h2, huber_of_one_lt h3,
    abs_of_nonpos (by norm_num : (4 : ℝ) / 5 - 2 ≤ 0)]
  norm_num

lemma huberRisk_at_zero : huberRisk 0 = 11 / 20 := by
  have h1 : |(0 : ℝ)| ≤ 1 := by rw [abs_le]; constructor <;> norm_num
  have h2 : |(0 : ℝ) - 1| ≤ 1 := by
    rw [abs_of_nonpos (by norm_num : (0 : ℝ) - 1 ≤ 0)]; norm_num
  have h3 : (1 : ℝ) < |(0 : ℝ) - 2| := by
    rw [abs_of_nonpos (by norm_num : (0 : ℝ) - 2 ≤ 0)]; norm_num
  rw [huberRisk, huber_of_abs_le h1, huber_of_abs_le h2, huber_of_one_lt h3,
    abs_of_nonpos (by norm_num : (0 : ℝ) - 2 ≤ 0)]
  norm_num

/-- **Theorem J (iii), global minimality.** `z = 4/5` minimizes the Huber
risk over all of `ℝ`: three tangent-line bounds combine with zero net
slope. -/
theorem huberRisk_argmin (z : ℝ) : huberRisk (4 / 5) ≤ huberRisk z := by
  have h1 : |(4 / 5 : ℝ)| ≤ 1 := by rw [abs_le]; constructor <;> norm_num
  have h2 : |(4 / 5 : ℝ) - 1| ≤ 1 := by rw [abs_le]; constructor <;> norm_num
  have ht1 := huber_tangent_le h1 z
  have ht2 := huber_tangent_le h2 (z - 1)
  have ht3 : -(z - 2) - 1 / 2 ≤ huber (z - 2) := by
    have ha := huber_ge_abs (z - 2)
    have hb := neg_le_abs (z - 2)
    linarith
  rw [huberRisk_at_min, huberRisk]
  linarith

/-- The Huber minimizer `4/5` is strictly closer to `1` than to `0` or
`2`: nearest-integer decoding returns `1`. -/
theorem huber_minimizer_rounds_to_one :
    |(4 / 5 : ℝ) - 1| < |(4 / 5 : ℝ) - 0| ∧
      |(4 / 5 : ℝ) - 1| < |(4 / 5 : ℝ) - 2| := by
  rw [abs_of_nonpos (by norm_num : (4 : ℝ) / 5 - 1 ≤ 0),
    abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 4 / 5 - 0),
    abs_of_nonpos (by norm_num : (4 : ℝ) / 5 - 2 ≤ 0)]
  constructor <;> norm_num

/-- The mode of the distribution is `0`: it strictly beats both other
values. -/
theorem huber_example_mode_is_zero : (7 / 20 : ℝ) < 2 / 5 ∧ (1 / 4 : ℝ) < 2 / 5 := by
  constructor <;> norm_num

/-- The mode-decode point `0` is strictly Huber-suboptimal. -/
theorem huberRisk_min_lt_at_mode : huberRisk (4 / 5) < huberRisk 0 := by
  rw [huberRisk_at_min, huberRisk_at_zero]; norm_num

/-- **Theorem J (iii), packaged.** For `P = (2/5, 7/20, 1/4)` on
`{0, 1, 2}` with Huber `δ = 1`: the global risk minimizer is `4/5`, which
rounds to `1`, yet the exact-match-optimal decode (the mode) is `0`. -/
theorem huber_decode_mode_mismatch :
    (∀ z : ℝ, huberRisk (4 / 5) ≤ huberRisk z) ∧
      (|(4 / 5 : ℝ) - 1| < |(4 / 5 : ℝ) - 0| ∧
        |(4 / 5 : ℝ) - 1| < |(4 / 5 : ℝ) - 2|) ∧
      ((7 / 20 : ℝ) < 2 / 5 ∧ (1 / 4 : ℝ) < 2 / 5) :=
  ⟨huberRisk_argmin, huber_minimizer_rounds_to_one, huber_example_mode_is_zero⟩

end TofyFormal
