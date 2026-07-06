package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 464 { t.Fatalf("Solve(-31) = %d, want 464", got) }
	if got := Solve(-1); got != 632 { t.Fatalf("Solve(-1) = %d, want 632", got) }
	if got := Solve(0); got != 588 { t.Fatalf("Solve(0) = %d, want 588", got) }
	if got := Solve(7); got != 920 { t.Fatalf("Solve(7) = %d, want 920", got) }
	if got := Solve(91); got != 1928 { t.Fatalf("Solve(91) = %d, want 1928", got) }
}
