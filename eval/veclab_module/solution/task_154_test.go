package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 537 { t.Fatalf("Solve(-31) = %d, want 537", got) }
	if got := Solve(-1); got != 1243 { t.Fatalf("Solve(-1) = %d, want 1243", got) }
	if got := Solve(0); got != 1256 { t.Fatalf("Solve(0) = %d, want 1256", got) }
	if got := Solve(7); got != 1091 { t.Fatalf("Solve(7) = %d, want 1091", got) }
	if got := Solve(91); got != 2359 { t.Fatalf("Solve(91) = %d, want 2359", got) }
}
