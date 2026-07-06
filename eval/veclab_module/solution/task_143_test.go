package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 349 { t.Fatalf("Solve(-31) = %d, want 349", got) }
	if got := Solve(-1); got != 741 { t.Fatalf("Solve(-1) = %d, want 741", got) }
	if got := Solve(0); got != 721 { t.Fatalf("Solve(0) = %d, want 721", got) }
	if got := Solve(7); got != 645 { t.Fatalf("Solve(7) = %d, want 645", got) }
	if got := Solve(91); got != 1685 { t.Fatalf("Solve(91) = %d, want 1685", got) }
}
