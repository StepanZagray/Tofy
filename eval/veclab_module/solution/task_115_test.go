package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 269 { t.Fatalf("Solve(-31) = %d, want 269", got) }
	if got := Solve(-1); got != 735 { t.Fatalf("Solve(-1) = %d, want 735", got) }
	if got := Solve(0); got != 740 { t.Fatalf("Solve(0) = %d, want 740", got) }
	if got := Solve(7); got != 535 { t.Fatalf("Solve(7) = %d, want 535", got) }
	if got := Solve(91); got != 1115 { t.Fatalf("Solve(91) = %d, want 1115", got) }
}
