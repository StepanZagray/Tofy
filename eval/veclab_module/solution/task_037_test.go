package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -295 { t.Fatalf("Solve(-31) = %d, want -295", got) }
	if got := Solve(-1); got != 369 { t.Fatalf("Solve(-1) = %d, want 369", got) }
	if got := Solve(0); got != 365 { t.Fatalf("Solve(0) = %d, want 365", got) }
	if got := Solve(7); got != 465 { t.Fatalf("Solve(7) = %d, want 465", got) }
	if got := Solve(91); got != 2113 { t.Fatalf("Solve(91) = %d, want 2113", got) }
}
