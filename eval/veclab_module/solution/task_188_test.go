package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 806 { t.Fatalf("Solve(-31) = %d, want 806", got) }
	if got := Solve(-1); got != 1384 { t.Fatalf("Solve(-1) = %d, want 1384", got) }
	if got := Solve(0); got != 1397 { t.Fatalf("Solve(0) = %d, want 1397", got) }
	if got := Solve(7); got != 1520 { t.Fatalf("Solve(7) = %d, want 1520", got) }
	if got := Solve(91); got != 3124 { t.Fatalf("Solve(91) = %d, want 3124", got) }
}
