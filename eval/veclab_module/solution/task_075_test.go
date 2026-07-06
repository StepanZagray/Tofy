package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 74 { t.Fatalf("Solve(-31) = %d, want 74", got) }
	if got := Solve(-1); got != 194 { t.Fatalf("Solve(-1) = %d, want 194", got) }
	if got := Solve(0); got != 206 { t.Fatalf("Solve(0) = %d, want 206", got) }
	if got := Solve(7); got != 290 { t.Fatalf("Solve(7) = %d, want 290", got) }
	if got := Solve(91); got != 594 { t.Fatalf("Solve(91) = %d, want 594", got) }
}
