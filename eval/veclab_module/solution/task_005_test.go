package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 256 { t.Fatalf("Solve(-31) = %d, want 256", got) }
	if got := Solve(-1); got != 628 { t.Fatalf("Solve(-1) = %d, want 628", got) }
	if got := Solve(0); got != 586 { t.Fatalf("Solve(0) = %d, want 586", got) }
	if got := Solve(7); got != 548 { t.Fatalf("Solve(7) = %d, want 548", got) }
	if got := Solve(91); got != 1068 { t.Fatalf("Solve(91) = %d, want 1068", got) }
}
