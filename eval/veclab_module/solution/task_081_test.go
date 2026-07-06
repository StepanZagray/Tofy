package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 84 { t.Fatalf("Solve(-31) = %d, want 84", got) }
	if got := Solve(-1); got != 494 { t.Fatalf("Solve(-1) = %d, want 494", got) }
	if got := Solve(0); got != 491 { t.Fatalf("Solve(0) = %d, want 491", got) }
	if got := Solve(7); got != 454 { t.Fatalf("Solve(7) = %d, want 454", got) }
	if got := Solve(91); got != 762 { t.Fatalf("Solve(91) = %d, want 762", got) }
}
