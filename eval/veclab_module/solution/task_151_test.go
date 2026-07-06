package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 823 { t.Fatalf("Solve(-31) = %d, want 823", got) }
	if got := Solve(-1); got != 1979 { t.Fatalf("Solve(-1) = %d, want 1979", got) }
	if got := Solve(0); got != 1941 { t.Fatalf("Solve(0) = %d, want 1941", got) }
	if got := Solve(7); got != 1867 { t.Fatalf("Solve(7) = %d, want 1867", got) }
	if got := Solve(91); got != 4451 { t.Fatalf("Solve(91) = %d, want 4451", got) }
}
