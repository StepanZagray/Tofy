package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -185 { t.Fatalf("Solve(-31) = %d, want -185", got) }
	if got := Solve(-1); got != 157 { t.Fatalf("Solve(-1) = %d, want 157", got) }
	if got := Solve(0); got != 162 { t.Fatalf("Solve(0) = %d, want 162", got) }
	if got := Solve(7); got != 181 { t.Fatalf("Solve(7) = %d, want 181", got) }
	if got := Solve(91); got != 433 { t.Fatalf("Solve(91) = %d, want 433", got) }
}
