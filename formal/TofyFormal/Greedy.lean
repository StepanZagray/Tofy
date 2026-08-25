import Mathlib

/-!
# Theorem I: greedy ordinal sufficiency and margin robustness

Greedy action selection only needs the *ordering* of Q-values: any
maximizer of a function with a strict unique maximum is that maximum
(ordinal sufficiency), a gap `γ` in the true Q-values survives any
uniform value error `ε` with `2ε < γ`, and hence a greedy policy on the
approximate values still picks the true optimal action.
-/

namespace TofyFormal

/-- **Theorem I (i), ordinal sufficiency.** If `astar` is the strict
unique maximizer of `q`, then every maximizer of `q` equals `astar`. -/
theorem every_maximizer_is_correct {A : Type*} {q : A → ℝ} {astar : A}
    (hstrict : ∀ a, a ≠ astar → q a < q astar)
    {ahat : A} (hmax : ∀ a, q a ≤ q ahat) : ahat = astar := by
  by_contra h
  exact absurd (hmax astar) (not_le.mpr (hstrict ahat h))

/-- **Theorem I (ii), margin robustness.** A gap of `γ` between the best
action and all others survives uniform value error `ε` when `2ε < γ`:
the approximate values still rank `astar` strictly first. -/
theorem ranking_survives_uniform_error {A : Type*} {Q Qhat : A → ℝ}
    {astar : A} {γ ε : ℝ}
    (hgap : ∀ a, a ≠ astar → Q a + γ ≤ Q astar)
    (herr : ∀ a, |Qhat a - Q a| ≤ ε) (hεγ : 2 * ε < γ) :
    ∀ a, a ≠ astar → Qhat a < Qhat astar := by
  intro a ha
  obtain ⟨h1, h2⟩ := abs_le.mp (herr a)
  obtain ⟨h3, h4⟩ := abs_le.mp (herr astar)
  have := hgap a ha
  linarith

/-- **Theorem I (iii).** Under a `γ`-gap and uniform error `ε` with
`2ε < γ`, any maximizer of the approximate values `Qhat` is the true
optimal action. -/
theorem greedy_on_approx_is_correct {A : Type*} {Q Qhat : A → ℝ}
    {astar : A} {γ ε : ℝ}
    (hgap : ∀ a, a ≠ astar → Q a + γ ≤ Q astar)
    (herr : ∀ a, |Qhat a - Q a| ≤ ε) (hεγ : 2 * ε < γ)
    {ahat : A} (hmax : ∀ a, Qhat a ≤ Qhat ahat) : ahat = astar :=
  every_maximizer_is_correct
    (ranking_survives_uniform_error hgap herr hεγ) hmax

end TofyFormal
