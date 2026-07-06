package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 174 { t.Fatalf("Solve(-31) = %d, want 174", got) }
	if got := Solve(-1); got != 616 { t.Fatalf("Solve(-1) = %d, want 616", got) }
	if got := Solve(0); got != 605 { t.Fatalf("Solve(0) = %d, want 605", got) }
	if got := Solve(7); got != 960 { t.Fatalf("Solve(7) = %d, want 960", got) }
	if got := Solve(91); got != 2460 { t.Fatalf("Solve(91) = %d, want 2460", got) }
}
