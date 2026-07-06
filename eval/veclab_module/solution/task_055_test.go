package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1279 { t.Fatalf("Solve(-31) = %d, want 1279", got) }
	if got := Solve(-1); got != 1789 { t.Fatalf("Solve(-1) = %d, want 1789", got) }
	if got := Solve(0); got != 1550 { t.Fatalf("Solve(0) = %d, want 1550", got) }
	if got := Solve(7); got != 1653 { t.Fatalf("Solve(7) = %d, want 1653", got) }
	if got := Solve(91); got != 3073 { t.Fatalf("Solve(91) = %d, want 3073", got) }
}
