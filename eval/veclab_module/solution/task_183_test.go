package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 558 { t.Fatalf("Solve(-31) = %d, want 558", got) }
	if got := Solve(-1); got != 1136 { t.Fatalf("Solve(-1) = %d, want 1136", got) }
	if got := Solve(0); got != 1151 { t.Fatalf("Solve(0) = %d, want 1151", got) }
	if got := Solve(7); got != 1512 { t.Fatalf("Solve(7) = %d, want 1512", got) }
	if got := Solve(91); got != 2772 { t.Fatalf("Solve(91) = %d, want 2772", got) }
}
