package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1192 { t.Fatalf("Solve(-31) = %d, want 1192", got) }
	if got := Solve(-1); got != 2006 { t.Fatalf("Solve(-1) = %d, want 2006", got) }
	if got := Solve(0); got != 1969 { t.Fatalf("Solve(0) = %d, want 1969", got) }
	if got := Solve(7); got != 2078 { t.Fatalf("Solve(7) = %d, want 2078", got) }
	if got := Solve(91); got != 4306 { t.Fatalf("Solve(91) = %d, want 4306", got) }
}
