package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -197 { t.Fatalf("Solve(-31) = %d, want -197", got) }
	if got := Solve(-1); got != 619 { t.Fatalf("Solve(-1) = %d, want 619", got) }
	if got := Solve(0); got != 595 { t.Fatalf("Solve(0) = %d, want 595", got) }
	if got := Solve(7); got != 683 { t.Fatalf("Solve(7) = %d, want 683", got) }
	if got := Solve(91); got != 2763 { t.Fatalf("Solve(91) = %d, want 2763", got) }
}
