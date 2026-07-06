package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 253 { t.Fatalf("Solve(-31) = %d, want 253", got) }
	if got := Solve(-1); got != 357 { t.Fatalf("Solve(-1) = %d, want 357", got) }
	if got := Solve(0); got != 377 { t.Fatalf("Solve(0) = %d, want 377", got) }
	if got := Solve(7); got != 325 { t.Fatalf("Solve(7) = %d, want 325", got) }
	if got := Solve(91); got != 533 { t.Fatalf("Solve(91) = %d, want 533", got) }
}
