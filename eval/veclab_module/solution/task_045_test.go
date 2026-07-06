package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -221 { t.Fatalf("Solve(-31) = %d, want -221", got) }
	if got := Solve(-1); got != 21 { t.Fatalf("Solve(-1) = %d, want 21", got) }
	if got := Solve(0); got != 28 { t.Fatalf("Solve(0) = %d, want 28", got) }
	if got := Solve(7); got != 93 { t.Fatalf("Solve(7) = %d, want 93", got) }
	if got := Solve(91); got != 873 { t.Fatalf("Solve(91) = %d, want 873", got) }
}
