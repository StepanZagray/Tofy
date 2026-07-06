package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 840 { t.Fatalf("Solve(-31) = %d, want 840", got) }
	if got := Solve(-1); got != 1714 { t.Fatalf("Solve(-1) = %d, want 1714", got) }
	if got := Solve(0); got != 1759 { t.Fatalf("Solve(0) = %d, want 1759", got) }
	if got := Solve(7); got != 1642 { t.Fatalf("Solve(7) = %d, want 1642", got) }
	if got := Solve(91); got != 3398 { t.Fatalf("Solve(91) = %d, want 3398", got) }
}
