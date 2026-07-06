package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 268 { t.Fatalf("Solve(-31) = %d, want 268", got) }
	if got := Solve(-1); got != 610 { t.Fatalf("Solve(-1) = %d, want 610", got) }
	if got := Solve(0); got != 613 { t.Fatalf("Solve(0) = %d, want 613", got) }
	if got := Solve(7); got != 570 { t.Fatalf("Solve(7) = %d, want 570", got) }
	if got := Solve(91); got != 1254 { t.Fatalf("Solve(91) = %d, want 1254", got) }
}
