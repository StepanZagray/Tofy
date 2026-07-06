package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 175 { t.Fatalf("Solve(-31) = %d, want 175", got) }
	if got := Solve(-1); got != 811 { t.Fatalf("Solve(-1) = %d, want 811", got) }
	if got := Solve(0); got != 781 { t.Fatalf("Solve(0) = %d, want 781", got) }
	if got := Solve(7); got != 1243 { t.Fatalf("Solve(7) = %d, want 1243", got) }
	if got := Solve(91); got != 3811 { t.Fatalf("Solve(91) = %d, want 3811", got) }
}
