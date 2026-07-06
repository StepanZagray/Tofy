package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 446 { t.Fatalf("Solve(-31) = %d, want 446", got) }
	if got := Solve(-1); got != 1220 { t.Fatalf("Solve(-1) = %d, want 1220", got) }
	if got := Solve(0); got != 1273 { t.Fatalf("Solve(0) = %d, want 1273", got) }
	if got := Solve(7); got != 1468 { t.Fatalf("Solve(7) = %d, want 1468", got) }
	if got := Solve(91); got != 3728 { t.Fatalf("Solve(91) = %d, want 3728", got) }
}
