package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 202 { t.Fatalf("Solve(-31) = %d, want 202", got) }
	if got := Solve(-1); got != 674 { t.Fatalf("Solve(-1) = %d, want 674", got) }
	if got := Solve(0); got != 702 { t.Fatalf("Solve(0) = %d, want 702", got) }
	if got := Solve(7); got != 962 { t.Fatalf("Solve(7) = %d, want 962", got) }
	if got := Solve(91); got != 2642 { t.Fatalf("Solve(91) = %d, want 2642", got) }
}
