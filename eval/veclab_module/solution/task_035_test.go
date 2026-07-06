package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 86 { t.Fatalf("Solve(-31) = %d, want 86", got) }
	if got := Solve(-1); got != 1242 { t.Fatalf("Solve(-1) = %d, want 1242", got) }
	if got := Solve(0); got != 1204 { t.Fatalf("Solve(0) = %d, want 1204", got) }
	if got := Solve(7); got != 1482 { t.Fatalf("Solve(7) = %d, want 1482", got) }
	if got := Solve(91); got != 4066 { t.Fatalf("Solve(91) = %d, want 4066", got) }
}
