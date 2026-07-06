package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 759 { t.Fatalf("Solve(-31) = %d, want 759", got) }
	if got := Solve(-1); got != 1227 { t.Fatalf("Solve(-1) = %d, want 1227", got) }
	if got := Solve(0); got != 1061 { t.Fatalf("Solve(0) = %d, want 1061", got) }
	if got := Solve(7); got != 1115 { t.Fatalf("Solve(7) = %d, want 1115", got) }
	if got := Solve(91); got != 2627 { t.Fatalf("Solve(91) = %d, want 2627", got) }
}
