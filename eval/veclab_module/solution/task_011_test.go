package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -49 { t.Fatalf("Solve(-31) = %d, want -49", got) }
	if got := Solve(-1); got != 117 { t.Fatalf("Solve(-1) = %d, want 117", got) }
	if got := Solve(0); got != 114 { t.Fatalf("Solve(0) = %d, want 114", got) }
	if got := Solve(7); got != 141 { t.Fatalf("Solve(7) = %d, want 141", got) }
	if got := Solve(91); got != 553 { t.Fatalf("Solve(91) = %d, want 553", got) }
}
