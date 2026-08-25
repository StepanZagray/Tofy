import Mathlib

/-!
# Theorem G: marginal-regularizer blindness

If the latent map `g` reads only the state component of a state-action
pair, the induced latent distribution depends only on the state marginal
of the joint data distribution.  Hence *any* regularizer defined on latent
distributions assigns the same value whether or not actions carry
information — such a regularizer is blind to action-dependence.  Stated
for `PMF` and for finite weight functions.
-/

namespace TofyFormal

/-- Pushing a joint distribution forward through a state-only latent map
factors through the state marginal. -/
theorem pushforward_state_latent_eq {S A Z : Type*}
    (μ : PMF (S × A)) (g : S → Z) :
    μ.map (fun p => g p.1) = (μ.map Prod.fst).map g := by
  rw [PMF.map_comp]
  rfl

/-- **Theorem G (PMF version).** Any regularizer on latent distributions
gives the same value on the joint-induced and marginal-induced latent
distributions: it cannot detect action-dependence. -/
theorem marginal_regularizer_blind {S A Z : Type*}
    (R : PMF Z → ℝ) (μ : PMF (S × A)) (g : S → Z) :
    R (μ.map (fun p => g p.1)) = R ((μ.map Prod.fst).map g) :=
  congrArg R (pushforward_state_latent_eq μ g)

/-- Finite-weight version of the pushforward identity: for each latent
value `z`, the joint weight of `{p | g p.1 = z}` equals the marginal
weight of `{s | g s = z}`. -/
theorem pushforward_weights_state_latent_eq {S A Z : Type*}
    [Fintype S] [Fintype A] [DecidableEq Z]
    (μ : S × A → ℝ) (g : S → Z) (z : Z) :
    ∑ p ∈ Finset.univ.filter (fun p : S × A => g p.1 = z), μ p
      = ∑ s ∈ Finset.univ.filter (fun s => g s = z), ∑ a, μ (s, a) := by
  classical
  rw [Finset.sum_filter, Finset.sum_filter, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun s _ => ?_
  by_cases h : g s = z
  · simp [h]
  · simp [h]

/-- **Theorem G (finite-weight version).** A regularizer applied to the
latent weight function sees only the state marginal. -/
theorem marginal_regularizer_blind_weights {S A Z : Type*}
    [Fintype S] [Fintype A] [DecidableEq Z]
    (R : (Z → ℝ) → ℝ) (μ : S × A → ℝ) (g : S → Z) :
    R (fun z => ∑ p ∈ Finset.univ.filter (fun p : S × A => g p.1 = z), μ p)
      = R (fun z => ∑ s ∈ Finset.univ.filter (fun s => g s = z), ∑ a, μ (s, a)) :=
  congrArg R (funext fun z => pushforward_weights_state_latent_eq μ g z)

end TofyFormal
