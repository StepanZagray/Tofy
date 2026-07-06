package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 380 { t.Fatalf("Solve(-31) = %d, want 380", got) }
	if got := Solve(-1); got != 524 { t.Fatalf("Solve(-1) = %d, want 524", got) }
	if got := Solve(0); got != 516 { t.Fatalf("Solve(0) = %d, want 516", got) }
	if got := Solve(7); got != 588 { t.Fatalf("Solve(7) = %d, want 588", got) }
	if got := Solve(91); got != 1324 { t.Fatalf("Solve(91) = %d, want 1324", got) }
}
