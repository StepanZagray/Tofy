package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 381 { t.Fatalf("Solve(-31) = %d, want 381", got) }
	if got := Solve(-1); got != 805 { t.Fatalf("Solve(-1) = %d, want 805", got) }
	if got := Solve(0); got != 777 { t.Fatalf("Solve(0) = %d, want 777", got) }
	if got := Solve(7); got != 901 { t.Fatalf("Solve(7) = %d, want 901", got) }
	if got := Solve(91); got != 2805 { t.Fatalf("Solve(91) = %d, want 2805", got) }
}
