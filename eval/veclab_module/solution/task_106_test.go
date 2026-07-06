package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 437 { t.Fatalf("Solve(-31) = %d, want 437", got) }
	if got := Solve(-1); got != 771 { t.Fatalf("Solve(-1) = %d, want 771", got) }
	if got := Solve(0); got != 1276 { t.Fatalf("Solve(0) = %d, want 1276", got) }
	if got := Solve(7); got != 1115 { t.Fatalf("Solve(7) = %d, want 1115", got) }
	if got := Solve(91); got != 3279 { t.Fatalf("Solve(91) = %d, want 3279", got) }
}
