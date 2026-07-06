package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 7 { t.Fatalf("Solve(-31) = %d, want 7", got) }
	if got := Solve(-1); got != 639 { t.Fatalf("Solve(-1) = %d, want 639", got) }
	if got := Solve(0); got != 659 { t.Fatalf("Solve(0) = %d, want 659", got) }
	if got := Solve(7); got != 799 { t.Fatalf("Solve(7) = %d, want 799", got) }
	if got := Solve(91); got != 2447 { t.Fatalf("Solve(91) = %d, want 2447", got) }
}
