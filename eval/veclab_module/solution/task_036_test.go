package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1053 { t.Fatalf("Solve(-31) = %d, want 1053", got) }
	if got := Solve(-1); got != 1807 { t.Fatalf("Solve(-1) = %d, want 1807", got) }
	if got := Solve(0); got != 1812 { t.Fatalf("Solve(0) = %d, want 1812", got) }
	if got := Solve(7); got != 1991 { t.Fatalf("Solve(7) = %d, want 1991", got) }
	if got := Solve(91); got != 4147 { t.Fatalf("Solve(91) = %d, want 4147", got) }
}
