package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 473 { t.Fatalf("Solve(-31) = %d, want 473", got) }
	if got := Solve(-1); got != 569 { t.Fatalf("Solve(-1) = %d, want 569", got) }
	if got := Solve(0); got != 553 { t.Fatalf("Solve(0) = %d, want 553", got) }
	if got := Solve(7); got != 953 { t.Fatalf("Solve(7) = %d, want 953", got) }
	if got := Solve(91); got != 2169 { t.Fatalf("Solve(91) = %d, want 2169", got) }
}
