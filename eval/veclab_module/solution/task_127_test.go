package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 163 { t.Fatalf("Solve(-31) = %d, want 163", got) }
	if got := Solve(-1); got != 9 { t.Fatalf("Solve(-1) = %d, want 9", got) }
	if got := Solve(0); got != 14 { t.Fatalf("Solve(0) = %d, want 14", got) }
	if got := Solve(7); got != 481 { t.Fatalf("Solve(7) = %d, want 481", got) }
	if got := Solve(91); got != 581 { t.Fatalf("Solve(91) = %d, want 581", got) }
}
