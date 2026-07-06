package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -691 { t.Fatalf("Solve(-31) = %d, want -691", got) }
	if got := Solve(-1); got != 189 { t.Fatalf("Solve(-1) = %d, want 189", got) }
	if got := Solve(0); got != 165 { t.Fatalf("Solve(0) = %d, want 165", got) }
	if got := Solve(7); got != 509 { t.Fatalf("Solve(7) = %d, want 509", got) }
	if got := Solve(91); got != 2525 { t.Fatalf("Solve(91) = %d, want 2525", got) }
}
