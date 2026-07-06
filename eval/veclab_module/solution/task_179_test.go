package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -155 { t.Fatalf("Solve(-31) = %d, want -155", got) }
	if got := Solve(-1); got != 49 { t.Fatalf("Solve(-1) = %d, want 49", got) }
	if got := Solve(0); got != 43 { t.Fatalf("Solve(0) = %d, want 43", got) }
	if got := Solve(7); got != 65 { t.Fatalf("Solve(7) = %d, want 65", got) }
	if got := Solve(91); got != 585 { t.Fatalf("Solve(91) = %d, want 585", got) }
}
