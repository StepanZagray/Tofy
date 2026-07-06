package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 589 { t.Fatalf("Solve(-31) = %d, want 589", got) }
	if got := Solve(-1); got != 921 { t.Fatalf("Solve(-1) = %d, want 921", got) }
	if got := Solve(0); got != 919 { t.Fatalf("Solve(0) = %d, want 919", got) }
	if got := Solve(7); got != 969 { t.Fatalf("Solve(7) = %d, want 969", got) }
	if got := Solve(91); got != 1793 { t.Fatalf("Solve(91) = %d, want 1793", got) }
}
