package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 34 { t.Fatalf("Solve(-31) = %d, want 34", got) }
	if got := Solve(-1); got != 300 { t.Fatalf("Solve(-1) = %d, want 300", got) }
	if got := Solve(0); got != 309 { t.Fatalf("Solve(0) = %d, want 309", got) }
	if got := Solve(7); got != 372 { t.Fatalf("Solve(7) = %d, want 372", got) }
	if got := Solve(91); got != 1128 { t.Fatalf("Solve(91) = %d, want 1128", got) }
}
