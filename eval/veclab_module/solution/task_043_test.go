package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 248 { t.Fatalf("Solve(-31) = %d, want 248", got) }
	if got := Solve(-1); got != 802 { t.Fatalf("Solve(-1) = %d, want 802", got) }
	if got := Solve(0); got != 823 { t.Fatalf("Solve(0) = %d, want 823", got) }
	if got := Solve(7); got != 906 { t.Fatalf("Solve(7) = %d, want 906", got) }
	if got := Solve(91); got != 2550 { t.Fatalf("Solve(91) = %d, want 2550", got) }
}
