package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 483 { t.Fatalf("Solve(-31) = %d, want 483", got) }
	if got := Solve(-1); got != 1105 { t.Fatalf("Solve(-1) = %d, want 1105", got) }
	if got := Solve(0); got != 1210 { t.Fatalf("Solve(0) = %d, want 1210", got) }
	if got := Solve(7); got != 1305 { t.Fatalf("Solve(7) = %d, want 1305", got) }
	if got := Solve(91); got != 3309 { t.Fatalf("Solve(91) = %d, want 3309", got) }
}
