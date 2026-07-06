package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 135 { t.Fatalf("Solve(-31) = %d, want 135", got) }
	if got := Solve(-1); got != 15 { t.Fatalf("Solve(-1) = %d, want 15", got) }
	if got := Solve(0); got != 3 { t.Fatalf("Solve(0) = %d, want 3", got) }
	if got := Solve(7); got != 47 { t.Fatalf("Solve(7) = %d, want 47", got) }
	if got := Solve(91); got != 767 { t.Fatalf("Solve(91) = %d, want 767", got) }
}
