package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -96 { t.Fatalf("Solve(-31) = %d, want -96", got) }
	if got := Solve(-1); got != 384 { t.Fatalf("Solve(-1) = %d, want 384", got) }
	if got := Solve(0); got != 400 { t.Fatalf("Solve(0) = %d, want 400", got) }
	if got := Solve(7); got != 512 { t.Fatalf("Solve(7) = %d, want 512", got) }
	if got := Solve(91); got != 1984 { t.Fatalf("Solve(91) = %d, want 1984", got) }
}
