package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 45 { t.Fatalf("Solve(-31) = %d, want 45", got) }
	if got := Solve(-1); got != 75 { t.Fatalf("Solve(-1) = %d, want 75", got) }
	if got := Solve(0); got != 78 { t.Fatalf("Solve(0) = %d, want 78", got) }
	if got := Solve(7); got != 419 { t.Fatalf("Solve(7) = %d, want 419", got) }
	if got := Solve(91); got != 703 { t.Fatalf("Solve(91) = %d, want 703", got) }
}
