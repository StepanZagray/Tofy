package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -112 { t.Fatalf("Solve(-31) = %d, want -112", got) }
	if got := Solve(-1); got != 196 { t.Fatalf("Solve(-1) = %d, want 196", got) }
	if got := Solve(0); got != 222 { t.Fatalf("Solve(0) = %d, want 222", got) }
	if got := Solve(7); got != 276 { t.Fatalf("Solve(7) = %d, want 276", got) }
	if got := Solve(91); got != 1132 { t.Fatalf("Solve(91) = %d, want 1132", got) }
}
