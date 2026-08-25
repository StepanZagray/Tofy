import Mathlib

/-!
# Theorem C: reactive-policy impossibility

If two goals demand distinct uniquely-optimal actions at the same
observation, then no goal-blind (reactive) policy `π : O → A` is optimal
for both goals: the policy must condition on the goal.
-/

namespace TofyFormal

/-- **Theorem C.** Suppose at observation `o` goal `g₁` has unique optimal
action `a₁` and goal `g₂` has unique optimal action `a₂`, with `a₁ ≠ a₂`.
Then no reactive policy `π : O → A` maximizes `Q g o ·` for both goals. -/
theorem no_reactive_policy_optimal_for_both
    {O A G : Type*} (Q : G → O → A → ℝ) {g₁ g₂ : G} {o : O} {a₁ a₂ : A}
    (hne : a₁ ≠ a₂)
    (h₁ : ∀ a, a ≠ a₁ → Q g₁ o a < Q g₁ o a₁)
    (h₂ : ∀ a, a ≠ a₂ → Q g₂ o a < Q g₂ o a₂)
    (π : O → A) :
    ¬ ((∀ a, Q g₁ o a ≤ Q g₁ o (π o)) ∧ (∀ a, Q g₂ o a ≤ Q g₂ o (π o))) := by
  rintro ⟨hopt₁, hopt₂⟩
  have e₁ : π o = a₁ := by
    by_contra h
    exact absurd (hopt₁ a₁) (not_le.mpr (h₁ (π o) h))
  have e₂ : π o = a₂ := by
    by_contra h
    exact absurd (hopt₂ a₂) (not_le.mpr (h₂ (π o) h))
  exact hne (e₁.symm.trans e₂)

end TofyFormal
