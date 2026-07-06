package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 6 { t.Fatalf("Solve(-31) = %d, want 6", got) }
	if got := Solve(-1); got != 550 { t.Fatalf("Solve(-1) = %d, want 550", got) }
	if got := Solve(0); got != 566 { t.Fatalf("Solve(0) = %d, want 566", got) }
	if got := Solve(7); got != 934 { t.Fatalf("Solve(7) = %d, want 934", got) }
	if got := Solve(91); got != 2278 { t.Fatalf("Solve(91) = %d, want 2278", got) }
}
