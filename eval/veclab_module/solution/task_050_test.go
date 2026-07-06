package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -309 { t.Fatalf("Solve(-31) = %d, want -309", got) }
	if got := Solve(-1); got != 81 { t.Fatalf("Solve(-1) = %d, want 81", got) }
	if got := Solve(0); got != 166 { t.Fatalf("Solve(0) = %d, want 166", got) }
	if got := Solve(7); got != 249 { t.Fatalf("Solve(7) = %d, want 249", got) }
	if got := Solve(91); got != 1285 { t.Fatalf("Solve(91) = %d, want 1285", got) }
}
