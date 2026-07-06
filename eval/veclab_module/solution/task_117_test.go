package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 834 { t.Fatalf("Solve(-31) = %d, want 834", got) }
	if got := Solve(-1); got != 1662 { t.Fatalf("Solve(-1) = %d, want 1662", got) }
	if got := Solve(0); got != 1688 { t.Fatalf("Solve(0) = %d, want 1688", got) }
	if got := Solve(7); got != 1838 { t.Fatalf("Solve(7) = %d, want 1838", got) }
	if got := Solve(91); got != 4006 { t.Fatalf("Solve(91) = %d, want 4006", got) }
}
