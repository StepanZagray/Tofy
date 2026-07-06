package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1089 { t.Fatalf("Solve(-31) = %d, want 1089", got) }
	if got := Solve(-1); got != 1831 { t.Fatalf("Solve(-1) = %d, want 1831", got) }
	if got := Solve(0); got != 2242 { t.Fatalf("Solve(0) = %d, want 2242", got) }
	if got := Solve(7); got != 2079 { t.Fatalf("Solve(7) = %d, want 2079", got) }
	if got := Solve(91); got != 4755 { t.Fatalf("Solve(91) = %d, want 4755", got) }
}
