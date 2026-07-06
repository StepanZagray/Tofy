package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 744 { t.Fatalf("Solve(-31) = %d, want 744", got) }
	if got := Solve(-1); got != 930 { t.Fatalf("Solve(-1) = %d, want 930", got) }
	if got := Solve(0); got != 951 { t.Fatalf("Solve(0) = %d, want 951", got) }
	if got := Solve(7); got != 842 { t.Fatalf("Solve(7) = %d, want 842", got) }
	if got := Solve(91); got != 1966 { t.Fatalf("Solve(91) = %d, want 1966", got) }
}
