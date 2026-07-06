package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -264 { t.Fatalf("Solve(-31) = %d, want -264", got) }
	if got := Solve(-1); got != 456 { t.Fatalf("Solve(-1) = %d, want 456", got) }
	if got := Solve(0); got != 464 { t.Fatalf("Solve(0) = %d, want 464", got) }
	if got := Solve(7); got != 520 { t.Fatalf("Solve(7) = %d, want 520", got) }
	if got := Solve(91); got != 2664 { t.Fatalf("Solve(91) = %d, want 2664", got) }
}
