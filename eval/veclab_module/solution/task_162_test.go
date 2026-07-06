package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 11 { t.Fatalf("Solve(-31) = %d, want 11", got) }
	if got := Solve(-1); got != 499 { t.Fatalf("Solve(-1) = %d, want 499", got) }
	if got := Solve(0); got != 399 { t.Fatalf("Solve(0) = %d, want 399", got) }
	if got := Solve(7); got != 403 { t.Fatalf("Solve(7) = %d, want 403", got) }
	if got := Solve(91); got != 611 { t.Fatalf("Solve(91) = %d, want 611", got) }
}
