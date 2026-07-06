package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 104 { t.Fatalf("Solve(-31) = %d, want 104", got) }
	if got := Solve(-1); got != 376 { t.Fatalf("Solve(-1) = %d, want 376", got) }
	if got := Solve(0); got != 368 { t.Fatalf("Solve(0) = %d, want 368", got) }
	if got := Solve(7); got != 440 { t.Fatalf("Solve(7) = %d, want 440", got) }
	if got := Solve(91); got != 1048 { t.Fatalf("Solve(91) = %d, want 1048", got) }
}
