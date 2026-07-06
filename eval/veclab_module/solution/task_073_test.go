package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 599 { t.Fatalf("Solve(-31) = %d, want 599", got) }
	if got := Solve(-1); got != 1083 { t.Fatalf("Solve(-1) = %d, want 1083", got) }
	if got := Solve(0); got != 1029 { t.Fatalf("Solve(0) = %d, want 1029", got) }
	if got := Solve(7); got != 1451 { t.Fatalf("Solve(7) = %d, want 1451", got) }
	if got := Solve(91); got != 2307 { t.Fatalf("Solve(91) = %d, want 2307", got) }
}
