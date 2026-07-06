package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 562 { t.Fatalf("Solve(-31) = %d, want 562", got) }
	if got := Solve(-1); got != 878 { t.Fatalf("Solve(-1) = %d, want 878", got) }
	if got := Solve(0); got != 888 { t.Fatalf("Solve(0) = %d, want 888", got) }
	if got := Solve(7); got != 958 { t.Fatalf("Solve(7) = %d, want 958", got) }
	if got := Solve(91); got != 1782 { t.Fatalf("Solve(91) = %d, want 1782", got) }
}
