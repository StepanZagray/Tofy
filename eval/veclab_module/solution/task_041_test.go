package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 844 { t.Fatalf("Solve(-31) = %d, want 844", got) }
	if got := Solve(-1); got != 2006 { t.Fatalf("Solve(-1) = %d, want 2006", got) }
	if got := Solve(0); got != 1995 { t.Fatalf("Solve(0) = %d, want 1995", got) }
	if got := Solve(7); got != 2238 { t.Fatalf("Solve(7) = %d, want 2238", got) }
	if got := Solve(91); got != 4410 { t.Fatalf("Solve(91) = %d, want 4410", got) }
}
