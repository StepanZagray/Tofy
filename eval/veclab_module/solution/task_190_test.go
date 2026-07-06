package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 25 { t.Fatalf("Solve(-31) = %d, want 25", got) }
	if got := Solve(-1); got != 267 { t.Fatalf("Solve(-1) = %d, want 267", got) }
	if got := Solve(0); got != 258 { t.Fatalf("Solve(0) = %d, want 258", got) }
	if got := Solve(7); got != 323 { t.Fatalf("Solve(7) = %d, want 323", got) }
	if got := Solve(91); got != 1111 { t.Fatalf("Solve(91) = %d, want 1111", got) }
}
