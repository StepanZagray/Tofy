package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 593 { t.Fatalf("Solve(-31) = %d, want 593", got) }
	if got := Solve(-1); got != 1517 { t.Fatalf("Solve(-1) = %d, want 1517", got) }
	if got := Solve(0); got != 1523 { t.Fatalf("Solve(0) = %d, want 1523", got) }
	if got := Solve(7); got != 1757 { t.Fatalf("Solve(7) = %d, want 1757", got) }
	if got := Solve(91); got != 4133 { t.Fatalf("Solve(91) = %d, want 4133", got) }
}
