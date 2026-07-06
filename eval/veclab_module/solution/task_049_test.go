package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 614 { t.Fatalf("Solve(-31) = %d, want 614", got) }
	if got := Solve(-1); got != 1026 { t.Fatalf("Solve(-1) = %d, want 1026", got) }
	if got := Solve(0); got != 1076 { t.Fatalf("Solve(0) = %d, want 1076", got) }
	if got := Solve(7); got != 1458 { t.Fatalf("Solve(7) = %d, want 1458", got) }
	if got := Solve(91); got != 2954 { t.Fatalf("Solve(91) = %d, want 2954", got) }
}
