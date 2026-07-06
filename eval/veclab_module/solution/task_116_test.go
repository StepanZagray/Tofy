package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -204 { t.Fatalf("Solve(-31) = %d, want -204", got) }
	if got := Solve(-1); got != 248 { t.Fatalf("Solve(-1) = %d, want 248", got) }
	if got := Solve(0); got != 250 { t.Fatalf("Solve(0) = %d, want 250", got) }
	if got := Solve(7); got != 136 { t.Fatalf("Solve(7) = %d, want 136", got) }
	if got := Solve(91); got != 32 { t.Fatalf("Solve(91) = %d, want 32", got) }
}
