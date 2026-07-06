package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -116 { t.Fatalf("Solve(-31) = %d, want -116", got) }
	if got := Solve(-1); got != 468 { t.Fatalf("Solve(-1) = %d, want 468", got) }
	if got := Solve(0); got != 312 { t.Fatalf("Solve(0) = %d, want 312", got) }
	if got := Solve(7); got != 692 { t.Fatalf("Solve(7) = %d, want 692", got) }
	if got := Solve(91); got != 2052 { t.Fatalf("Solve(91) = %d, want 2052", got) }
}
