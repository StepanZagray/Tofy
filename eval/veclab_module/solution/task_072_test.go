package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 287 { t.Fatalf("Solve(-31) = %d, want 287", got) }
	if got := Solve(-1); got != 801 { t.Fatalf("Solve(-1) = %d, want 801", got) }
	if got := Solve(0); got != 972 { t.Fatalf("Solve(0) = %d, want 972", got) }
	if got := Solve(7); got != 1097 { t.Fatalf("Solve(7) = %d, want 1097", got) }
	if got := Solve(91); got != 2573 { t.Fatalf("Solve(91) = %d, want 2573", got) }
}
