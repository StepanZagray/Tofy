package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -30 { t.Fatalf("Solve(-31) = %d, want -30", got) }
	if got := Solve(-1); got != 92 { t.Fatalf("Solve(-1) = %d, want 92", got) }
	if got := Solve(0); got != 95 { t.Fatalf("Solve(0) = %d, want 95", got) }
	if got := Solve(7); got != 116 { t.Fatalf("Solve(7) = %d, want 116", got) }
	if got := Solve(91); got != 368 { t.Fatalf("Solve(91) = %d, want 368", got) }
}
