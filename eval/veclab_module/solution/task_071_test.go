package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -459 { t.Fatalf("Solve(-31) = %d, want -459", got) }
	if got := Solve(-1); got != 229 { t.Fatalf("Solve(-1) = %d, want 229", got) }
	if got := Solve(0); got != 285 { t.Fatalf("Solve(0) = %d, want 285", got) }
	if got := Solve(7); got != 421 { t.Fatalf("Solve(7) = %d, want 421", got) }
	if got := Solve(91); got != 2437 { t.Fatalf("Solve(91) = %d, want 2437", got) }
}
