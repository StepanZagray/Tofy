package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1377 { t.Fatalf("Solve(-31) = %d, want 1377", got) }
	if got := Solve(-1); got != 2037 { t.Fatalf("Solve(-1) = %d, want 2037", got) }
	if got := Solve(0); got != 1947 { t.Fatalf("Solve(0) = %d, want 1947", got) }
	if got := Solve(7); got != 2085 { t.Fatalf("Solve(7) = %d, want 2085", got) }
	if got := Solve(91); got != 4077 { t.Fatalf("Solve(91) = %d, want 4077", got) }
}
