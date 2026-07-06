package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -6 { t.Fatalf("Solve(-31) = %d, want -6", got) }
	if got := Solve(-1); got != 520 { t.Fatalf("Solve(-1) = %d, want 520", got) }
	if got := Solve(0); got != 563 { t.Fatalf("Solve(0) = %d, want 563", got) }
	if got := Solve(7); got != 848 { t.Fatalf("Solve(7) = %d, want 848", got) }
	if got := Solve(91); got != 2756 { t.Fatalf("Solve(91) = %d, want 2756", got) }
}
