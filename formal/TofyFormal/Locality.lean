import Mathlib

/-!
# Theorem F: light cone of local update rules

`LocalAt r F` says the update rule `F` computes its output at `u` from the
input restricted to the closed ball of radius `r` around `u`.  Composition
adds radii, `d`-fold iteration reaches radius `d * r` (the light cone), and
consequently no `R`-local rule can compute a target whose value at `u`
depends on the input beyond radius `R`.
-/

namespace TofyFormal

variable {V X : Type*} [PseudoMetricSpace V]

/-- An update rule `F` is `r`-local at every site: its output at `u`
depends only on the input within distance `r` of `u`. -/
def LocalAt (r : ℝ) (F : (V → X) → V → X) : Prop :=
  ∀ x y u, (∀ v, dist v u ≤ r → x v = y v) → F x u = F y u

/-- **Theorem F (i).** Composing an `rF`-local rule after an `rG`-local
rule yields an `(rF + rG)`-local rule. -/
theorem LocalAt.comp {rF rG : ℝ} {F G : (V → X) → V → X}
    (hF : LocalAt rF F) (hG : LocalAt rG G) : LocalAt (rF + rG) (F ∘ G) := by
  intro x y u hagree
  refine hF (G x) (G y) u fun v hv => hG x y v fun w hw => hagree w ?_
  calc dist w u ≤ dist w v + dist v u := dist_triangle _ _ _
    _ ≤ rG + rF := add_le_add hw hv
    _ = rF + rG := add_comm _ _

/-- **Theorem F (ii), light cone.** Iterating an `r`-local rule `d` times
yields a `(d * r)`-local rule: information propagates at most `r` per
step. -/
theorem LocalAt.iterate {r : ℝ} {F : (V → X) → V → X}
    (hF : LocalAt r F) (d : ℕ) : LocalAt (d * r) F^[d] := by
  induction d with
  | zero =>
    intro x y u hagree
    simpa using hagree u (by simp)
  | succ n ih =>
    have harith : ((n + 1 : ℕ) : ℝ) * r = r + (n : ℕ) * r := by
      push_cast; ring
    rw [harith, Function.iterate_succ']
    exact hF.comp ih

/-- An `R`-local rule cannot distinguish inputs that agree within radius
`R` of `u`. -/
theorem LocalAt.eq_of_agree {R : ℝ} {F : (V → X) → V → X}
    (hF : LocalAt R F) {x y : V → X} {u : V}
    (hagree : ∀ v, dist v u ≤ R → x v = y v) : F x u = F y u :=
  hF x y u hagree

/-- **Theorem F (iii), impossibility.** If the target `T` takes different
values at `u` on two inputs that agree within radius `R` of `u`, then no
`R`-local rule agrees with `T` on both inputs at `u`. -/
theorem LocalAt.not_computes {R : ℝ} {F T : (V → X) → V → X}
    (hF : LocalAt R F) {x y : V → X} {u : V}
    (hagree : ∀ v, dist v u ≤ R → x v = y v) (hT : T x u ≠ T y u) :
    ¬(F x u = T x u ∧ F y u = T y u) := by
  rintro ⟨h1, h2⟩
  exact hT (by rw [← h1, ← h2]; exact hF x y u hagree)

end TofyFormal
