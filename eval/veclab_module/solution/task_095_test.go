package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -185 { t.Fatalf("Solve(-31) = %d, want -185", got) }
	if got := Solve(-1); got != 303 { t.Fatalf("Solve(-1) = %d, want 303", got) }
	if got := Solve(0); got != 723 { t.Fatalf("Solve(0) = %d, want 723", got) }
	if got := Solve(7); got != 591 { t.Fatalf("Solve(7) = %d, want 591", got) }
	if got := Solve(91); got != 2559 { t.Fatalf("Solve(91) = %d, want 2559", got) }
}
