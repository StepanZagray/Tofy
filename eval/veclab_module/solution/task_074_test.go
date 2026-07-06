package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -224 { t.Fatalf("Solve(-31) = %d, want -224", got) }
	if got := Solve(-1); got != 50 { t.Fatalf("Solve(-1) = %d, want 50", got) }
	if got := Solve(0); got != 11 { t.Fatalf("Solve(0) = %d, want 11", got) }
	if got := Solve(7); got != 458 { t.Fatalf("Solve(7) = %d, want 458", got) }
	if got := Solve(91); got != 1278 { t.Fatalf("Solve(91) = %d, want 1278", got) }
}
