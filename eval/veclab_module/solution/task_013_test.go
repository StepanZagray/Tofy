package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1714 { t.Fatalf("Solve(-31) = %d, want 1714", got) }
	if got := Solve(-1); got != 2114 { t.Fatalf("Solve(-1) = %d, want 2114", got) }
	if got := Solve(0); got != 2474 { t.Fatalf("Solve(0) = %d, want 2474", got) }
	if got := Solve(7); got != 2306 { t.Fatalf("Solve(7) = %d, want 2306", got) }
	if got := Solve(91); got != 4386 { t.Fatalf("Solve(91) = %d, want 4386", got) }
}
