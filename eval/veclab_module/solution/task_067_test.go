package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 361 { t.Fatalf("Solve(-31) = %d, want 361", got) }
	if got := Solve(-1); got != 935 { t.Fatalf("Solve(-1) = %d, want 935", got) }
	if got := Solve(0); got != 952 { t.Fatalf("Solve(0) = %d, want 952", got) }
	if got := Solve(7); got != 815 { t.Fatalf("Solve(7) = %d, want 815", got) }
	if got := Solve(91); got != 2115 { t.Fatalf("Solve(91) = %d, want 2115", got) }
}
