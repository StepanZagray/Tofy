package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1488 { t.Fatalf("Solve(-31) = %d, want 1488", got) }
	if got := Solve(-1); got != 2430 { t.Fatalf("Solve(-1) = %d, want 2430", got) }
	if got := Solve(0); got != 2387 { t.Fatalf("Solve(0) = %d, want 2387", got) }
	if got := Solve(7); got != 2566 { t.Fatalf("Solve(7) = %d, want 2566", got) }
	if got := Solve(91); got != 4994 { t.Fatalf("Solve(91) = %d, want 4994", got) }
}
