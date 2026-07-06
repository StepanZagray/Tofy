package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -564 { t.Fatalf("Solve(-31) = %d, want -564", got) }
	if got := Solve(-1); got != 116 { t.Fatalf("Solve(-1) = %d, want 116", got) }
	if got := Solve(0); got != 80 { t.Fatalf("Solve(0) = %d, want 80", got) }
	if got := Solve(7); got != 276 { t.Fatalf("Solve(7) = %d, want 276", got) }
	if got := Solve(91); got != 2660 { t.Fatalf("Solve(91) = %d, want 2660", got) }
}
