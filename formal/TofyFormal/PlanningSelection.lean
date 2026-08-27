import Mathlib

/-!
# Deterministic selection-charge bounds

Selecting the largest noisy score can select a candidate whose true value is
below the true finite maximum.  For scores `v i + e i`, the shortfall is at
most twice the largest absolute error.  The factor two is tight even with two
candidates and a tie in the noisy scores.

This module does not formalize the ADR's distributional selection-charge
identity for independent sub-Gaussian increments: that requires a probability
model and expectations.  The correlated-score case remains OPEN as well.
-/

namespace TofyFormal

open Finset

/-- The maximum of a real-valued function over a nonempty finite set. -/
noncomputable def finiteMaximum {ι : Type*} (I : Finset ι) (f : ι → ℝ)
    (hI : I.Nonempty) : ℝ :=
  (I.image f).max' (by
    obtain ⟨i, hi⟩ := hI
    exact ⟨f i, Finset.mem_image.mpr ⟨i, hi, rfl⟩⟩)

/-- The largest absolute score error over a nonempty finite candidate set. -/
noncomputable def errorRadius {ι : Type*} (I : Finset ι) (e : ι → ℝ)
    (hI : I.Nonempty) : ℝ :=
  finiteMaximum I (fun i => |e i|) hI

/-- A score maximizer loses at most twice a uniform absolute score error in
true value. -/
theorem true_value_le_selected_of_argmax {ι : Type*} {I : Finset ι}
    {v e : ι → ℝ} {chosen : ι}
    (hchosen : chosen ∈ I)
    (hmax : ∀ i ∈ I, v i + e i ≤ v chosen + e chosen)
    {ε : ℝ} (herr : ∀ i ∈ I, |e i| ≤ ε) :
    ∀ i ∈ I, v i ≤ v chosen + 2 * ε := by
  intro i hi
  have hilo := (abs_le.mp (herr i hi)).1
  have hchosenhi := (abs_le.mp (herr chosen hchosen)).2
  have hscore := hmax i hi
  linarith

/-- **Selection-charge bound.** If `chosen` maximizes the noisy scores over a
nonempty finite candidate set, its true value is within twice the maximum
absolute score error of the true finite maximum. -/
theorem finite_argmax_selection_bound {ι : Type*} (I : Finset ι)
    {v e : ι → ℝ} (hI : I.Nonempty) {chosen : ι}
    (hchosen : chosen ∈ I)
    (hmax : ∀ i ∈ I, v i + e i ≤ v chosen + e chosen) :
    finiteMaximum I v hI ≤ v chosen + 2 * errorRadius I e hI := by
  have herr : ∀ j ∈ I, |e j| ≤ errorRadius I e hI := by
    intro j hj
    unfold errorRadius finiteMaximum
    apply Finset.le_max'
    exact Finset.mem_image.mpr ⟨j, hj, rfl⟩
  apply Finset.max'_le
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hx
  exact true_value_le_selected_of_argmax hchosen hmax herr i hi

/-- Enlarging a finite candidate set cannot decrease the largest absolute
score error. -/
theorem errorRadius_mono {ι : Type*} {I J : Finset ι} {e : ι → ℝ}
    (hI : I.Nonempty) (hJ : J.Nonempty) (hsub : I ⊆ J) :
    errorRadius I e hI ≤ errorRadius J e hJ := by
  unfold errorRadius finiteMaximum
  apply Finset.max'_le
  intro x hx
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hx
  apply Finset.le_max'
  exact Finset.mem_image.mpr ⟨i, hsub hi, rfl⟩

/-- The factor two in `finite_argmax_selection_bound` is tight: two
candidates with opposite errors can tie in noisy score while their true
values differ by exactly `2 * ε`. -/
theorem selection_charge_factor_two_tight (ε : ℝ) (hε : 0 ≤ ε) :
    ∃ (v e : Fin 2 → ℝ) (chosen best : Fin 2),
      (∀ i, v i + e i ≤ v chosen + e chosen) ∧
      errorRadius Finset.univ e Finset.univ_nonempty = ε ∧
      v best = v chosen + 2 * errorRadius Finset.univ e Finset.univ_nonempty := by
  let v : Fin 2 → ℝ := fun i => if i = 0 then 0 else 2 * ε
  let e : Fin 2 → ℝ := fun i => if i = 0 then ε else -ε
  have hRadius : errorRadius Finset.univ e Finset.univ_nonempty = ε := by
    apply le_antisymm
    · unfold errorRadius finiteMaximum
      apply Finset.max'_le
      intro x hx
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hx
      fin_cases i <;> simp [e, abs_neg, abs_of_nonneg hε]
    · unfold errorRadius finiteMaximum
      apply Finset.le_max'
      exact Finset.mem_image.mpr ⟨0, Finset.mem_univ _, by
        simp [e, abs_of_nonneg hε]⟩
  refine ⟨v, e, 0, 1, ?_, hRadius, ?_⟩
  · intro i
    fin_cases i
    · simp [v, e]
    · simp only [v, e]
      norm_num
      linarith
  · rw [hRadius]
    simp only [v]
    norm_num

end TofyFormal
