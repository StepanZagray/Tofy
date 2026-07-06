package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 228 { t.Fatalf("Solve(-31) = %d, want 228", got) }
	if got := Solve(-1); got != 498 { t.Fatalf("Solve(-1) = %d, want 498", got) }
	if got := Solve(0); got != 267 { t.Fatalf("Solve(0) = %d, want 267", got) }
	if got := Solve(7); got != 330 { t.Fatalf("Solve(7) = %d, want 330", got) }
	if got := Solve(91); got != 1086 { t.Fatalf("Solve(91) = %d, want 1086", got) }
}
