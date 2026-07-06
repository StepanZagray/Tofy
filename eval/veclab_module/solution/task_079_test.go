package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -150 { t.Fatalf("Solve(-31) = %d, want -150", got) }
	if got := Solve(-1); got != 148 { t.Fatalf("Solve(-1) = %d, want 148", got) }
	if got := Solve(0); got != 135 { t.Fatalf("Solve(0) = %d, want 135", got) }
	if got := Solve(7); got != 380 { t.Fatalf("Solve(7) = %d, want 380", got) }
	if got := Solve(91); got != 1336 { t.Fatalf("Solve(91) = %d, want 1336", got) }
}
