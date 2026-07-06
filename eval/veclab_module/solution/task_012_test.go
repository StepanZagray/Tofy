package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 847 { t.Fatalf("Solve(-31) = %d, want 847", got) }
	if got := Solve(-1); got != 1777 { t.Fatalf("Solve(-1) = %d, want 1777", got) }
	if got := Solve(0); got != 1740 { t.Fatalf("Solve(0) = %d, want 1740", got) }
	if got := Solve(7); got != 1817 { t.Fatalf("Solve(7) = %d, want 1817", got) }
	if got := Solve(91); got != 4253 { t.Fatalf("Solve(91) = %d, want 4253", got) }
}
