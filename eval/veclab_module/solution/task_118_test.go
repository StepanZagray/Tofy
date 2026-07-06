package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -618 { t.Fatalf("Solve(-31) = %d, want -618", got) }
	if got := Solve(-1); got != 96 { t.Fatalf("Solve(-1) = %d, want 96", got) }
	if got := Solve(0); got != 11 { t.Fatalf("Solve(0) = %d, want 11", got) }
	if got := Solve(7); got != 184 { t.Fatalf("Solve(7) = %d, want 184", got) }
	if got := Solve(91); got != 1940 { t.Fatalf("Solve(91) = %d, want 1940", got) }
}
