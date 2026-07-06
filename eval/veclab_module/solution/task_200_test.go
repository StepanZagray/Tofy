package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 144 { t.Fatalf("Solve(-31) = %d, want 144", got) }
	if got := Solve(-1); got != 366 { t.Fatalf("Solve(-1) = %d, want 366", got) }
	if got := Solve(0); got != 671 { t.Fatalf("Solve(0) = %d, want 671", got) }
	if got := Solve(7); got != 534 { t.Fatalf("Solve(7) = %d, want 534", got) }
	if got := Solve(91); got != 2186 { t.Fatalf("Solve(91) = %d, want 2186", got) }
}
