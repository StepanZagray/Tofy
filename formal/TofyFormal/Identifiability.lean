import Mathlib

/-!
# Theorem A: counterfactual non-identifiability on finite branches

If a state-action pair `p₀` is outside the observed data `D`, then two
transition maps can agree with the ground truth everywhere on `D` yet
disagree at `p₀`: the data never determines off-support counterfactuals.
Conversely, if `D` covers every branch, agreement on `D` pins the map down.
-/

namespace TofyFormal

variable {S A Y : Type*}

/-- **Theorem A (negative direction).** With at least two possible outcomes,
any transition map `F` and any unobserved pair `p₀ ∉ D` admit two maps
`F₁, F₂` that both agree with `F` on all of `D` but differ at `p₀`. -/
theorem exists_agree_on_data_ne_off_support
    [DecidableEq S] [DecidableEq A] [Fintype Y] (hY : 2 ≤ Fintype.card Y)
    (D : Set (S × A)) (F : S × A → Y) {p₀ : S × A} (hp₀ : p₀ ∉ D) :
    ∃ F₁ F₂ : S × A → Y,
      (∀ p ∈ D, F₁ p = F p ∧ F₂ p = F p) ∧ F₁ p₀ ≠ F₂ p₀ := by
  have : Nontrivial Y := Fintype.one_lt_card_iff_nontrivial.mp hY
  obtain ⟨y, hy⟩ := exists_ne (F p₀)
  refine ⟨F, Function.update F p₀ y, fun p hp => ⟨rfl, ?_⟩, ?_⟩
  · refine Function.update_of_ne (fun h => ?_) _ _
    exact hp₀ (h ▸ hp)
  · rw [Function.update_self]
    exact hy.symm

/-- **Theorem A (corollary).** Agreement on the observed data does not
determine the transition: as soon as one branch is unobserved, "agrees on
`D`" does not imply equality of transition maps. -/
theorem not_identifiable_of_missing_branch
    [DecidableEq S] [DecidableEq A] [Fintype Y] (hY : 2 ≤ Fintype.card Y)
    (D : Set (S × A)) {p₀ : S × A} (hp₀ : p₀ ∉ D) :
    ¬ ∀ F₁ F₂ : S × A → Y, (∀ p ∈ D, F₁ p = F₂ p) → F₁ = F₂ := by
  intro hid
  have : Nontrivial Y := Fintype.one_lt_card_iff_nontrivial.mp hY
  obtain ⟨y₀⟩ := (inferInstance : Nonempty Y)
  obtain ⟨F₁, F₂, hagree, hne⟩ :=
    exists_agree_on_data_ne_off_support hY D (fun _ => y₀) hp₀
  refine hne (congrFun (hid F₁ F₂ fun p hp => ?_) p₀)
  exact (hagree p hp).1.trans (hagree p hp).2.symm

/-- **Theorem A (positive direction).** Under full branch coverage
(`D` contains every state-action pair), agreement on `D` identifies the
transition map. -/
theorem eq_of_agree_on_full_coverage
    (D : Set (S × A)) (hD : ∀ p, p ∈ D)
    {F₁ F₂ : S × A → Y} (h : ∀ p ∈ D, F₁ p = F₂ p) : F₁ = F₂ :=
  funext fun p => h p (hD p)

end TofyFormal
