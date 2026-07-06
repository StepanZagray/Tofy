package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 623 { t.Fatalf("Solve(-31) = %d, want 623", got) }
	if got := Solve(-1); got != 1429 { t.Fatalf("Solve(-1) = %d, want 1429", got) }
	if got := Solve(0); got != 1456 { t.Fatalf("Solve(0) = %d, want 1456", got) }
	if got := Solve(7); got != 1645 { t.Fatalf("Solve(7) = %d, want 1645", got) }
	if got := Solve(91); got != 3913 { t.Fatalf("Solve(91) = %d, want 3913", got) }
}
