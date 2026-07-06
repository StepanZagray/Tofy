package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 455 { t.Fatalf("Solve(-31) = %d, want 455", got) }
	if got := Solve(-1); got != 901 { t.Fatalf("Solve(-1) = %d, want 901", got) }
	if got := Solve(0); got != 918 { t.Fatalf("Solve(0) = %d, want 918", got) }
	if got := Solve(7); got != 1085 { t.Fatalf("Solve(7) = %d, want 1085", got) }
	if got := Solve(91); got != 2337 { t.Fatalf("Solve(91) = %d, want 2337", got) }
}
