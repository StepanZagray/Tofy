package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 0 { t.Fatalf("Solve(-31) = %d, want 0", got) }
	if got := Solve(-1); got != 68 { t.Fatalf("Solve(-1) = %d, want 68", got) }
	if got := Solve(0); got != 70 { t.Fatalf("Solve(0) = %d, want 70", got) }
	if got := Solve(7); got != 84 { t.Fatalf("Solve(7) = %d, want 84", got) }
	if got := Solve(91); got != 252 { t.Fatalf("Solve(91) = %d, want 252", got) }
}
