package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 218 { t.Fatalf("Solve(-31) = %d, want 218", got) }
	if got := Solve(-1); got != 626 { t.Fatalf("Solve(-1) = %d, want 626", got) }
	if got := Solve(0); got != 614 { t.Fatalf("Solve(0) = %d, want 614", got) }
	if got := Solve(7); got != 658 { t.Fatalf("Solve(7) = %d, want 658", got) }
	if got := Solve(91); got != 1698 { t.Fatalf("Solve(91) = %d, want 1698", got) }
}
