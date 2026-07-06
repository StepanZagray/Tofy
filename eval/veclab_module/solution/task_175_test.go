package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1260 { t.Fatalf("Solve(-31) = %d, want 1260", got) }
	if got := Solve(-1); got != 2016 { t.Fatalf("Solve(-1) = %d, want 2016", got) }
	if got := Solve(0); got != 1990 { t.Fatalf("Solve(0) = %d, want 1990", got) }
	if got := Solve(7); got != 2064 { t.Fatalf("Solve(7) = %d, want 2064", got) }
	if got := Solve(91); got != 4232 { t.Fatalf("Solve(91) = %d, want 4232", got) }
}
