package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -614 { t.Fatalf("Solve(-31) = %d, want -614", got) }
	if got := Solve(-1); got != 164 { t.Fatalf("Solve(-1) = %d, want 164", got) }
	if got := Solve(0); got != 157 { t.Fatalf("Solve(0) = %d, want 157", got) }
	if got := Solve(7); got != 364 { t.Fatalf("Solve(7) = %d, want 364", got) }
	if got := Solve(91); got != 2464 { t.Fatalf("Solve(91) = %d, want 2464", got) }
}
