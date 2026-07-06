package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 311 { t.Fatalf("Solve(-31) = %d, want 311", got) }
	if got := Solve(-1); got != 581 { t.Fatalf("Solve(-1) = %d, want 581", got) }
	if got := Solve(0); got != 606 { t.Fatalf("Solve(0) = %d, want 606", got) }
	if got := Solve(7); got != 525 { t.Fatalf("Solve(7) = %d, want 525", got) }
	if got := Solve(91); got != 1497 { t.Fatalf("Solve(91) = %d, want 1497", got) }
}
