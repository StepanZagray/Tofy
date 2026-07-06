package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 191 { t.Fatalf("Solve(-31) = %d, want 191", got) }
	if got := Solve(-1); got != 625 { t.Fatalf("Solve(-1) = %d, want 625", got) }
	if got := Solve(0); got != 614 { t.Fatalf("Solve(0) = %d, want 614", got) }
	if got := Solve(7); got != 777 { t.Fatalf("Solve(7) = %d, want 777", got) }
	if got := Solve(91); got != 2997 { t.Fatalf("Solve(91) = %d, want 2997", got) }
}
