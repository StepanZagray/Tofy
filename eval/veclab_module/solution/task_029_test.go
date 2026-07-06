package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -25 { t.Fatalf("Solve(-31) = %d, want -25", got) }
	if got := Solve(-1); got != 35 { t.Fatalf("Solve(-1) = %d, want 35", got) }
	if got := Solve(0); got != 33 { t.Fatalf("Solve(0) = %d, want 33", got) }
	if got := Solve(7); got != 19 { t.Fatalf("Solve(7) = %d, want 19", got) }
	if got := Solve(91); got != 251 { t.Fatalf("Solve(91) = %d, want 251", got) }
}
