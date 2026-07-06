package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 625 { t.Fatalf("Solve(-31) = %d, want 625", got) }
	if got := Solve(-1); got != 973 { t.Fatalf("Solve(-1) = %d, want 973", got) }
	if got := Solve(0); got != 963 { t.Fatalf("Solve(0) = %d, want 963", got) }
	if got := Solve(7); got != 1117 { t.Fatalf("Solve(7) = %d, want 1117", got) }
	if got := Solve(91); got != 2245 { t.Fatalf("Solve(91) = %d, want 2245", got) }
}
