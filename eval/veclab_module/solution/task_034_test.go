package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 279 { t.Fatalf("Solve(-31) = %d, want 279", got) }
	if got := Solve(-1); got != 731 { t.Fatalf("Solve(-1) = %d, want 731", got) }
	if got := Solve(0); got != 733 { t.Fatalf("Solve(0) = %d, want 733", got) }
	if got := Solve(7); got != 747 { t.Fatalf("Solve(7) = %d, want 747", got) }
	if got := Solve(91); got != 1267 { t.Fatalf("Solve(91) = %d, want 1267", got) }
}
