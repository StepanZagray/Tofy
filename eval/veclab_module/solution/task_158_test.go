package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -130 { t.Fatalf("Solve(-31) = %d, want -130", got) }
	if got := Solve(-1); got != 526 { t.Fatalf("Solve(-1) = %d, want 526", got) }
	if got := Solve(0); got != 534 { t.Fatalf("Solve(0) = %d, want 534", got) }
	if got := Solve(7); got != 974 { t.Fatalf("Solve(7) = %d, want 974", got) }
	if got := Solve(91); got != 3054 { t.Fatalf("Solve(91) = %d, want 3054", got) }
}
