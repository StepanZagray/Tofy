package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -285 { t.Fatalf("Solve(-31) = %d, want -285", got) }
	if got := Solve(-1); got != 7 { t.Fatalf("Solve(-1) = %d, want 7", got) }
	if got := Solve(0); got != 21 { t.Fatalf("Solve(0) = %d, want 21", got) }
	if got := Solve(7); got != 247 { t.Fatalf("Solve(7) = %d, want 247", got) }
	if got := Solve(91); got != 1295 { t.Fatalf("Solve(91) = %d, want 1295", got) }
}
