package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 262 { t.Fatalf("Solve(-31) = %d, want 262", got) }
	if got := Solve(-1); got != 588 { t.Fatalf("Solve(-1) = %d, want 588", got) }
	if got := Solve(0); got != 593 { t.Fatalf("Solve(0) = %d, want 593", got) }
	if got := Solve(7); got != 932 { t.Fatalf("Solve(7) = %d, want 932", got) }
	if got := Solve(91); got != 1592 { t.Fatalf("Solve(91) = %d, want 1592", got) }
}
