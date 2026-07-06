package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -117 { t.Fatalf("Solve(-31) = %d, want -117", got) }
	if got := Solve(-1); got != 95 { t.Fatalf("Solve(-1) = %d, want 95", got) }
	if got := Solve(0); got != 85 { t.Fatalf("Solve(0) = %d, want 85", got) }
	if got := Solve(7); got != 175 { t.Fatalf("Solve(7) = %d, want 175", got) }
	if got := Solve(91); got != 695 { t.Fatalf("Solve(91) = %d, want 695", got) }
}
