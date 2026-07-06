package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1805 { t.Fatalf("Solve(-31) = %d, want 1805", got) }
	if got := Solve(-1); got != 2757 { t.Fatalf("Solve(-1) = %d, want 2757", got) }
	if got := Solve(0); got != 2729 { t.Fatalf("Solve(0) = %d, want 2729", got) }
	if got := Solve(7); got != 3045 { t.Fatalf("Solve(7) = %d, want 3045", got) }
	if got := Solve(91); got != 5301 { t.Fatalf("Solve(91) = %d, want 5301", got) }
}
