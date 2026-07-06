package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 914 { t.Fatalf("Solve(-31) = %d, want 914", got) }
	if got := Solve(-1); got != 1358 { t.Fatalf("Solve(-1) = %d, want 1358", got) }
	if got := Solve(0); got != 1344 { t.Fatalf("Solve(0) = %d, want 1344", got) }
	if got := Solve(7); got != 1470 { t.Fatalf("Solve(7) = %d, want 1470", got) }
	if got := Solve(91); got != 2630 { t.Fatalf("Solve(91) = %d, want 2630", got) }
}
