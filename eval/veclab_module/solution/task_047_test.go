package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1666 { t.Fatalf("Solve(-31) = %d, want 1666", got) }
	if got := Solve(-1); got != 2650 { t.Fatalf("Solve(-1) = %d, want 2650", got) }
	if got := Solve(0); got != 2662 { t.Fatalf("Solve(0) = %d, want 2662", got) }
	if got := Solve(7); got != 2746 { t.Fatalf("Solve(7) = %d, want 2746", got) }
	if got := Solve(91); got != 5226 { t.Fatalf("Solve(91) = %d, want 5226", got) }
}
