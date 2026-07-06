package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 823 { t.Fatalf("Solve(-31) = %d, want 823", got) }
	if got := Solve(-1); got != 1691 { t.Fatalf("Solve(-1) = %d, want 1691", got) }
	if got := Solve(0); got != 1773 { t.Fatalf("Solve(0) = %d, want 1773", got) }
	if got := Solve(7); got != 1611 { t.Fatalf("Solve(7) = %d, want 1611", got) }
	if got := Solve(91); got != 3763 { t.Fatalf("Solve(91) = %d, want 3763", got) }
}
