package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -313 { t.Fatalf("Solve(-31) = %d, want -313", got) }
	if got := Solve(-1); got != 399 { t.Fatalf("Solve(-1) = %d, want 399", got) }
	if got := Solve(0); got != 363 { t.Fatalf("Solve(0) = %d, want 363", got) }
	if got := Solve(7); got != 687 { t.Fatalf("Solve(7) = %d, want 687", got) }
	if got := Solve(91); got != 2975 { t.Fatalf("Solve(91) = %d, want 2975", got) }
}
