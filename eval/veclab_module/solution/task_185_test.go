package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 212 { t.Fatalf("Solve(-31) = %d, want 212", got) }
	if got := Solve(-1); got != 334 { t.Fatalf("Solve(-1) = %d, want 334", got) }
	if got := Solve(0); got != 369 { t.Fatalf("Solve(0) = %d, want 369", got) }
	if got := Solve(7); got != 278 { t.Fatalf("Solve(7) = %d, want 278", got) }
	if got := Solve(91); got != 698 { t.Fatalf("Solve(91) = %d, want 698", got) }
}
