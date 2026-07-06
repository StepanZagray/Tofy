package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -198 { t.Fatalf("Solve(-31) = %d, want -198", got) }
	if got := Solve(-1); got != 754 { t.Fatalf("Solve(-1) = %d, want 754", got) }
	if got := Solve(0); got != 662 { t.Fatalf("Solve(0) = %d, want 662", got) }
	if got := Solve(7); got != 978 { t.Fatalf("Solve(7) = %d, want 978", got) }
	if got := Solve(91); got != 3202 { t.Fatalf("Solve(91) = %d, want 3202", got) }
}
