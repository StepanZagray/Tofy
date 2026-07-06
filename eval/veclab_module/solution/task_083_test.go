package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 864 { t.Fatalf("Solve(-31) = %d, want 864", got) }
	if got := Solve(-1); got != 1492 { t.Fatalf("Solve(-1) = %d, want 1492", got) }
	if got := Solve(0); got != 1466 { t.Fatalf("Solve(0) = %d, want 1466", got) }
	if got := Solve(7); got != 1572 { t.Fatalf("Solve(7) = %d, want 1572", got) }
	if got := Solve(91); got != 3564 { t.Fatalf("Solve(91) = %d, want 3564", got) }
}
