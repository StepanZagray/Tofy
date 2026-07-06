package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 314 { t.Fatalf("Solve(-31) = %d, want 314", got) }
	if got := Solve(-1); got != 864 { t.Fatalf("Solve(-1) = %d, want 864", got) }
	if got := Solve(0); got != 1165 { t.Fatalf("Solve(0) = %d, want 1165", got) }
	if got := Solve(7); got != 1032 { t.Fatalf("Solve(7) = %d, want 1032", got) }
	if got := Solve(91); got != 2636 { t.Fatalf("Solve(91) = %d, want 2636", got) }
}
