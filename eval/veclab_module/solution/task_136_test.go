package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1140 { t.Fatalf("Solve(-31) = %d, want 1140", got) }
	if got := Solve(-1); got != 1640 { t.Fatalf("Solve(-1) = %d, want 1640", got) }
	if got := Solve(0); got != 1606 { t.Fatalf("Solve(0) = %d, want 1606", got) }
	if got := Solve(7); got != 1752 { t.Fatalf("Solve(7) = %d, want 1752", got) }
	if got := Solve(91); got != 3296 { t.Fatalf("Solve(91) = %d, want 3296", got) }
}
