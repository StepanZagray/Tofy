package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 299 { t.Fatalf("Solve(-31) = %d, want 299", got) }
	if got := Solve(-1); got != 537 { t.Fatalf("Solve(-1) = %d, want 537", got) }
	if got := Solve(0); got != 530 { t.Fatalf("Solve(0) = %d, want 530", got) }
	if got := Solve(7); got != 545 { t.Fatalf("Solve(7) = %d, want 545", got) }
	if got := Solve(91); got != 1173 { t.Fatalf("Solve(91) = %d, want 1173", got) }
}
