package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 826 { t.Fatalf("Solve(-31) = %d, want 826", got) }
	if got := Solve(-1); got != 1366 { t.Fatalf("Solve(-1) = %d, want 1366", got) }
	if got := Solve(0); got != 1352 { t.Fatalf("Solve(0) = %d, want 1352", got) }
	if got := Solve(7); got != 1478 { t.Fatalf("Solve(7) = %d, want 1478", got) }
	if got := Solve(91); got != 2990 { t.Fatalf("Solve(91) = %d, want 2990", got) }
}
