package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -512 { t.Fatalf("Solve(-31) = %d, want -512", got) }
	if got := Solve(-1); got != 510 { t.Fatalf("Solve(-1) = %d, want 510", got) }
	if got := Solve(0); got != 405 { t.Fatalf("Solve(0) = %d, want 405", got) }
	if got := Solve(7); got != 294 { t.Fatalf("Solve(7) = %d, want 294", got) }
	if got := Solve(91); got != 2050 { t.Fatalf("Solve(91) = %d, want 2050", got) }
}
