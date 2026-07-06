package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 39 { t.Fatalf("Solve(-31) = %d, want 39", got) }
	if got := Solve(-1); got != 123 { t.Fatalf("Solve(-1) = %d, want 123", got) }
	if got := Solve(0); got != 121 { t.Fatalf("Solve(0) = %d, want 121", got) }
	if got := Solve(7); got != 107 { t.Fatalf("Solve(7) = %d, want 107", got) }
	if got := Solve(91); got != 307 { t.Fatalf("Solve(91) = %d, want 307", got) }
}
