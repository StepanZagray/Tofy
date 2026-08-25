import Mathlib

/-!
# Theorem H: separation loss forces action-dependent latents

The hinge separation loss `sepLoss m za zb = max 0 (m - dist za zb)`
vanishes only when the two latents are at distance at least `m`; in
particular they must differ.  Consequently an action-independent
displacement map pays exactly the full margin `m > 0` on any pair it is
asked to separate.
-/

namespace TofyFormal

variable {V : Type*} [PseudoMetricSpace V]

/-- Hinge separation loss with margin `m`. -/
noncomputable def sepLoss (m : ℝ) (za zb : V) : ℝ := max 0 (m - dist za zb)

/-- On identical latents the separation loss is the full margin. -/
theorem sepLoss_self {m : ℝ} (hm : 0 ≤ m) (z : V) : sepLoss m z z = m := by
  rw [sepLoss, dist_self, sub_zero, max_eq_right hm]

/-- Zero separation loss forces distance at least the margin. -/
theorem margin_le_dist_of_sepLoss_eq_zero {m : ℝ} {za zb : V}
    (h : sepLoss m za zb = 0) : m ≤ dist za zb := by
  have := le_max_right (0 : ℝ) (m - dist za zb)
  rw [sepLoss] at h
  linarith [h ▸ this]

/-- **Theorem H.** With positive margin, zero separation loss forces the
latents apart. -/
theorem ne_of_sepLoss_eq_zero {m : ℝ} (hm : 0 < m) {za zb : V}
    (h : sepLoss m za zb = 0) : za ≠ zb := by
  intro he
  rw [he, sepLoss_self hm.le] at h
  exact hm.ne' h

/-- An action-independent displacement map pays exactly the margin on any
pair of actions it maps identically. -/
theorem sepLoss_eq_margin_of_action_independent {A : Type*} {Δ : A → V}
    {m : ℝ} (hm : 0 ≤ m) {a b : A} (hΔ : Δ a = Δ b) :
    sepLoss m (Δ a) (Δ b) = m := by
  rw [hΔ, sepLoss_self hm]

/-- **Theorem H (corollary).** With positive margin, an
action-independent displacement map cannot achieve zero separation loss:
its loss on such a pair is nonzero (indeed equal to `m`). -/
theorem sepLoss_ne_zero_of_action_independent {A : Type*} {Δ : A → V}
    {m : ℝ} (hm : 0 < m) {a b : A} (hΔ : Δ a = Δ b) :
    sepLoss m (Δ a) (Δ b) ≠ 0 := by
  rw [sepLoss_eq_margin_of_action_independent hm.le hΔ]
  exact hm.ne'

end TofyFormal
