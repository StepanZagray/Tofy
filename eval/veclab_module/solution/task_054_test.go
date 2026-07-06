package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 721 { t.Fatalf("Solve(-31) = %d, want 721", got) }
	if got := Solve(-1); got != 1389 { t.Fatalf("Solve(-1) = %d, want 1389", got) }
	if got := Solve(0); got != 1403 { t.Fatalf("Solve(0) = %d, want 1403", got) }
	if got := Solve(7); got != 1437 { t.Fatalf("Solve(7) = %d, want 1437", got) }
	if got := Solve(91); got != 3413 { t.Fatalf("Solve(91) = %d, want 3413", got) }
}
