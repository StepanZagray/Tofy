package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -160 { t.Fatalf("Solve(-31) = %d, want -160", got) }
	if got := Solve(-1); got != 134 { t.Fatalf("Solve(-1) = %d, want 134", got) }
	if got := Solve(0); got != 133 { t.Fatalf("Solve(0) = %d, want 133", got) }
	if got := Solve(7); got != 254 { t.Fatalf("Solve(7) = %d, want 254", got) }
	if got := Solve(91); got != 498 { t.Fatalf("Solve(91) = %d, want 498", got) }
}
