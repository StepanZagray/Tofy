package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 592 { t.Fatalf("Solve(-31) = %d, want 592", got) }
	if got := Solve(-1); got != 1748 { t.Fatalf("Solve(-1) = %d, want 1748", got) }
	if got := Solve(0); got != 1778 { t.Fatalf("Solve(0) = %d, want 1778", got) }
	if got := Solve(7); got != 1988 { t.Fatalf("Solve(7) = %d, want 1988", got) }
	if got := Solve(91); got != 4508 { t.Fatalf("Solve(91) = %d, want 4508", got) }
}
