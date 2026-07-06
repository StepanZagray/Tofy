package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 151 { t.Fatalf("Solve(-31) = %d, want 151", got) }
	if got := Solve(-1); got != 527 { t.Fatalf("Solve(-1) = %d, want 527", got) }
	if got := Solve(0); got != 539 { t.Fatalf("Solve(0) = %d, want 539", got) }
	if got := Solve(7); got != 623 { t.Fatalf("Solve(7) = %d, want 623", got) }
	if got := Solve(91); got != 1631 { t.Fatalf("Solve(91) = %d, want 1631", got) }
}
