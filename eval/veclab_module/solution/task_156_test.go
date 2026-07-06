package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 73 { t.Fatalf("Solve(-31) = %d, want 73", got) }
	if got := Solve(-1); got != 223 { t.Fatalf("Solve(-1) = %d, want 223", got) }
	if got := Solve(0); got != 292 { t.Fatalf("Solve(0) = %d, want 292", got) }
	if got := Solve(7); got != 263 { t.Fatalf("Solve(7) = %d, want 263", got) }
	if got := Solve(91); got != 747 { t.Fatalf("Solve(91) = %d, want 747", got) }
}
