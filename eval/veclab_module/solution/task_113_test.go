package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -452 { t.Fatalf("Solve(-31) = %d, want -452", got) }
	if got := Solve(-1); got != 62 { t.Fatalf("Solve(-1) = %d, want 62", got) }
	if got := Solve(0); got != 47 { t.Fatalf("Solve(0) = %d, want 47", got) }
	if got := Solve(7); got != 422 { t.Fatalf("Solve(7) = %d, want 422", got) }
	if got := Solve(91); got != 1562 { t.Fatalf("Solve(91) = %d, want 1562", got) }
}
