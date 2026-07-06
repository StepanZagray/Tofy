package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 564 { t.Fatalf("Solve(-31) = %d, want 564", got) }
	if got := Solve(-1); got != 1386 { t.Fatalf("Solve(-1) = %d, want 1386", got) }
	if got := Solve(0); got != 1405 { t.Fatalf("Solve(0) = %d, want 1405", got) }
	if got := Solve(7); got != 1442 { t.Fatalf("Solve(7) = %d, want 1442", got) }
	if got := Solve(91); got != 3374 { t.Fatalf("Solve(91) = %d, want 3374", got) }
}
