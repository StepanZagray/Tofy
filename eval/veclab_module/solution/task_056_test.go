package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 135 { t.Fatalf("Solve(-31) = %d, want 135", got) }
	if got := Solve(-1); got != 367 { t.Fatalf("Solve(-1) = %d, want 367", got) }
	if got := Solve(0); got != 283 { t.Fatalf("Solve(0) = %d, want 283", got) }
	if got := Solve(7); got != 719 { t.Fatalf("Solve(7) = %d, want 719", got) }
	if got := Solve(91); got != 1759 { t.Fatalf("Solve(91) = %d, want 1759", got) }
}
