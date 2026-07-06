package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 220 { t.Fatalf("Solve(-31) = %d, want 220", got) }
	if got := Solve(-1); got != 614 { t.Fatalf("Solve(-1) = %d, want 614", got) }
	if got := Solve(0); got != 627 { t.Fatalf("Solve(0) = %d, want 627", got) }
	if got := Solve(7); got != 702 { t.Fatalf("Solve(7) = %d, want 702", got) }
	if got := Solve(91); got != 1562 { t.Fatalf("Solve(91) = %d, want 1562", got) }
}
