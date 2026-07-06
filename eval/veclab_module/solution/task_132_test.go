package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 99 { t.Fatalf("Solve(-31) = %d, want 99", got) }
	if got := Solve(-1); got != 273 { t.Fatalf("Solve(-1) = %d, want 273", got) }
	if got := Solve(0); got != 280 { t.Fatalf("Solve(0) = %d, want 280", got) }
	if got := Solve(7); got != 473 { t.Fatalf("Solve(7) = %d, want 473", got) }
	if got := Solve(91); got != 1237 { t.Fatalf("Solve(91) = %d, want 1237", got) }
}
