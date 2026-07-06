package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 141 { t.Fatalf("Solve(-31) = %d, want 141", got) }
	if got := Solve(-1); got != 39 { t.Fatalf("Solve(-1) = %d, want 39", got) }
	if got := Solve(0); got != 40 { t.Fatalf("Solve(0) = %d, want 40", got) }
	if got := Solve(7); got != 79 { t.Fatalf("Solve(7) = %d, want 79", got) }
	if got := Solve(91); got != 755 { t.Fatalf("Solve(91) = %d, want 755", got) }
}
