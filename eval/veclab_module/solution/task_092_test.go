package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -120 { t.Fatalf("Solve(-31) = %d, want -120", got) }
	if got := Solve(-1); got != 68 { t.Fatalf("Solve(-1) = %d, want 68", got) }
	if got := Solve(0); got != 78 { t.Fatalf("Solve(0) = %d, want 78", got) }
	if got := Solve(7); got != 116 { t.Fatalf("Solve(7) = %d, want 116", got) }
	if got := Solve(91); got != 620 { t.Fatalf("Solve(91) = %d, want 620", got) }
}
