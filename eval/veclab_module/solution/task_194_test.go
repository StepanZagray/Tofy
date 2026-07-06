package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -300 { t.Fatalf("Solve(-31) = %d, want -300", got) }
	if got := Solve(-1); got != 248 { t.Fatalf("Solve(-1) = %d, want 248", got) }
	if got := Solve(0); got != 234 { t.Fatalf("Solve(0) = %d, want 234", got) }
	if got := Solve(7); got != 104 { t.Fatalf("Solve(7) = %d, want 104", got) }
	if got := Solve(91); got != 1600 { t.Fatalf("Solve(91) = %d, want 1600", got) }
}
