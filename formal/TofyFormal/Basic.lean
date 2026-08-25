import Mathlib

/-!
# Tofy formalization scaffold

Finite-grid world-model setting shared by the theorems in this package:
states are functions from a finite cell set to a finite palette, actions
are a finite type, and the environment transition is a function
`S → A → S` observed only through a data distribution.
-/

namespace TofyFormal

/-- A finite grid-world signature: finitely many cells, palette colors,
and actions. -/
structure GridSig where
  Cell : Type
  Color : Type
  Action : Type
  [cellFin : Fintype Cell]
  [colorFin : Fintype Color]
  [actionFin : Fintype Action]
  [colorDec : DecidableEq Color]
  [cellDec : DecidableEq Cell]
  [actionDec : DecidableEq Action]

attribute [instance] GridSig.cellFin GridSig.colorFin GridSig.actionFin
attribute [instance] GridSig.colorDec GridSig.cellDec GridSig.actionDec

/-- A world state assigns a palette color to every cell. -/
def GridSig.State (G : GridSig) : Type := G.Cell → G.Color

instance (G : GridSig) : Fintype G.State := by
  unfold GridSig.State; infer_instance

instance (G : GridSig) : DecidableEq G.State := by
  unfold GridSig.State; infer_instance

end TofyFormal
