package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 620 { t.Fatalf("Solve(-31) = %d, want 620", got) }
	if got := Solve(-1); got != 1062 { t.Fatalf("Solve(-1) = %d, want 1062", got) }
	if got := Solve(0); got != 1043 { t.Fatalf("Solve(0) = %d, want 1043", got) }
	if got := Solve(7); got != 1438 { t.Fatalf("Solve(7) = %d, want 1438", got) }
	if got := Solve(91); got != 2898 { t.Fatalf("Solve(91) = %d, want 2898", got) }
}
