package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -382 { t.Fatalf("Solve(-31) = %d, want -382", got) }
	if got := Solve(-1); got != 238 { t.Fatalf("Solve(-1) = %d, want 238", got) }
	if got := Solve(0); got != 220 { t.Fatalf("Solve(0) = %d, want 220", got) }
	if got := Solve(7); got != 382 { t.Fatalf("Solve(7) = %d, want 382", got) }
	if got := Solve(91); got != 1494 { t.Fatalf("Solve(91) = %d, want 1494", got) }
}
