package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 493 { t.Fatalf("Solve(-31) = %d, want 493", got) }
	if got := Solve(-1); got != 315 { t.Fatalf("Solve(-1) = %d, want 315", got) }
	if got := Solve(0); got != 260 { t.Fatalf("Solve(0) = %d, want 260", got) }
	if got := Solve(7); got != 755 { t.Fatalf("Solve(7) = %d, want 755", got) }
	if got := Solve(91); got != 1159 { t.Fatalf("Solve(91) = %d, want 1159", got) }
}
