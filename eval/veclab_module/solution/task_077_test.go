package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 124 { t.Fatalf("Solve(-31) = %d, want 124", got) }
	if got := Solve(-1); got != 942 { t.Fatalf("Solve(-1) = %d, want 942", got) }
	if got := Solve(0); got != 837 { t.Fatalf("Solve(0) = %d, want 837", got) }
	if got := Solve(7); got != 1254 { t.Fatalf("Solve(7) = %d, want 1254", got) }
	if got := Solve(91); got != 2922 { t.Fatalf("Solve(91) = %d, want 2922", got) }
}
