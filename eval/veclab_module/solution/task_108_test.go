package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -255 { t.Fatalf("Solve(-31) = %d, want -255", got) }
	if got := Solve(-1); got != 411 { t.Fatalf("Solve(-1) = %d, want 411", got) }
	if got := Solve(0); got != 404 { t.Fatalf("Solve(0) = %d, want 404", got) }
	if got := Solve(7); got != 499 { t.Fatalf("Solve(7) = %d, want 499", got) }
	if got := Solve(91); got != 1335 { t.Fatalf("Solve(91) = %d, want 1335", got) }
}
