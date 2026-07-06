package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -38 { t.Fatalf("Solve(-31) = %d, want -38", got) }
	if got := Solve(-1); got != 232 { t.Fatalf("Solve(-1) = %d, want 232", got) }
	if got := Solve(0); got != 241 { t.Fatalf("Solve(0) = %d, want 241", got) }
	if got := Solve(7); got != 48 { t.Fatalf("Solve(7) = %d, want 48", got) }
	if got := Solve(91); got != 812 { t.Fatalf("Solve(91) = %d, want 812", got) }
}
