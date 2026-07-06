package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 2010 { t.Fatalf("Solve(-31) = %d, want 2010", got) }
	if got := Solve(-1); got != 2620 { t.Fatalf("Solve(-1) = %d, want 2620", got) }
	if got := Solve(0); got != 2649 { t.Fatalf("Solve(0) = %d, want 2649", got) }
	if got := Solve(7); got != 2836 { t.Fatalf("Solve(7) = %d, want 2836", got) }
	if got := Solve(91); got != 5512 { t.Fatalf("Solve(91) = %d, want 5512", got) }
}
