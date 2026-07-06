package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1649 { t.Fatalf("Solve(-31) = %d, want 1649", got) }
	if got := Solve(-1); got != 2435 { t.Fatalf("Solve(-1) = %d, want 2435", got) }
	if got := Solve(0); got != 2408 { t.Fatalf("Solve(0) = %d, want 2408", got) }
	if got := Solve(7); got != 2747 { t.Fatalf("Solve(7) = %d, want 2747", got) }
	if got := Solve(91); got != 4751 { t.Fatalf("Solve(91) = %d, want 4751", got) }
}
