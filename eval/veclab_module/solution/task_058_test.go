package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -122 { t.Fatalf("Solve(-31) = %d, want -122", got) }
	if got := Solve(-1); got != 122 { t.Fatalf("Solve(-1) = %d, want 122", got) }
	if got := Solve(0); got != 120 { t.Fatalf("Solve(0) = %d, want 120", got) }
	if got := Solve(7); got != 74 { t.Fatalf("Solve(7) = %d, want 74", got) }
	if got := Solve(91); got != 178 { t.Fatalf("Solve(91) = %d, want 178", got) }
}
