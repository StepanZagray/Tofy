package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1278 { t.Fatalf("Solve(-31) = %d, want 1278", got) }
	if got := Solve(-1); got != 1642 { t.Fatalf("Solve(-1) = %d, want 1642", got) }
	if got := Solve(0); got != 1556 { t.Fatalf("Solve(0) = %d, want 1556", got) }
	if got := Solve(7); got != 1978 { t.Fatalf("Solve(7) = %d, want 1978", got) }
	if got := Solve(91); got != 3650 { t.Fatalf("Solve(91) = %d, want 3650", got) }
}
