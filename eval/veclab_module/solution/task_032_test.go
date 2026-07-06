package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -115 { t.Fatalf("Solve(-31) = %d, want -115", got) }
	if got := Solve(-1); got != 365 { t.Fatalf("Solve(-1) = %d, want 365", got) }
	if got := Solve(0); got != 381 { t.Fatalf("Solve(0) = %d, want 381", got) }
	if got := Solve(7); got != 493 { t.Fatalf("Solve(7) = %d, want 493", got) }
	if got := Solve(91); got != 1837 { t.Fatalf("Solve(91) = %d, want 1837", got) }
}
