package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 608 { t.Fatalf("Solve(-31) = %d, want 608", got) }
	if got := Solve(-1); got != 934 { t.Fatalf("Solve(-1) = %d, want 934", got) }
	if got := Solve(0); got != 1105 { t.Fatalf("Solve(0) = %d, want 1105", got) }
	if got := Solve(7); got != 1070 { t.Fatalf("Solve(7) = %d, want 1070", got) }
	if got := Solve(91); got != 2362 { t.Fatalf("Solve(91) = %d, want 2362", got) }
}
