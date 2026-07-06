package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1078 { t.Fatalf("Solve(-31) = %d, want 1078", got) }
	if got := Solve(-1); got != 1996 { t.Fatalf("Solve(-1) = %d, want 1996", got) }
	if got := Solve(0); got != 1977 { t.Fatalf("Solve(0) = %d, want 1977", got) }
	if got := Solve(7); got != 1812 { t.Fatalf("Solve(7) = %d, want 1812", got) }
	if got := Solve(91); got != 3632 { t.Fatalf("Solve(91) = %d, want 3632", got) }
}
