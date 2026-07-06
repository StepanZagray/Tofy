package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 215 { t.Fatalf("Solve(-31) = %d, want 215", got) }
	if got := Solve(-1); got != 349 { t.Fatalf("Solve(-1) = %d, want 349", got) }
	if got := Solve(0); got != 330 { t.Fatalf("Solve(0) = %d, want 330", got) }
	if got := Solve(7); got != 741 { t.Fatalf("Solve(7) = %d, want 741", got) }
	if got := Solve(91); got != 1705 { t.Fatalf("Solve(91) = %d, want 1705", got) }
}
