package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1483 { t.Fatalf("Solve(-31) = %d, want 1483", got) }
	if got := Solve(-1); got != 2265 { t.Fatalf("Solve(-1) = %d, want 2265", got) }
	if got := Solve(0); got != 2290 { t.Fatalf("Solve(0) = %d, want 2290", got) }
	if got := Solve(7); got != 2433 { t.Fatalf("Solve(7) = %d, want 2433", got) }
	if got := Solve(91); got != 4573 { t.Fatalf("Solve(91) = %d, want 4573", got) }
}
