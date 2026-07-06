package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1781 { t.Fatalf("Solve(-31) = %d, want 1781", got) }
	if got := Solve(-1); got != 2463 { t.Fatalf("Solve(-1) = %d, want 2463", got) }
	if got := Solve(0); got != 2674 { t.Fatalf("Solve(0) = %d, want 2674", got) }
	if got := Solve(7); got != 2695 { t.Fatalf("Solve(7) = %d, want 2695", got) }
	if got := Solve(91); got != 5123 { t.Fatalf("Solve(91) = %d, want 5123", got) }
}
