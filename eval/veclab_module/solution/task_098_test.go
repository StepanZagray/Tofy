package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 152 { t.Fatalf("Solve(-31) = %d, want 152", got) }
	if got := Solve(-1); got != 118 { t.Fatalf("Solve(-1) = %d, want 118", got) }
	if got := Solve(0); got != 125 { t.Fatalf("Solve(0) = %d, want 125", got) }
	if got := Solve(7); got != 94 { t.Fatalf("Solve(7) = %d, want 94", got) }
	if got := Solve(91); got != 570 { t.Fatalf("Solve(91) = %d, want 570", got) }
}
