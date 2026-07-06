package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -299 { t.Fatalf("Solve(-31) = %d, want -299", got) }
	if got := Solve(-1); got != 47 { t.Fatalf("Solve(-1) = %d, want 47", got) }
	if got := Solve(0); got != 64 { t.Fatalf("Solve(0) = %d, want 64", got) }
	if got := Solve(7); got != 503 { t.Fatalf("Solve(7) = %d, want 503", got) }
	if got := Solve(91); got != 2267 { t.Fatalf("Solve(91) = %d, want 2267", got) }
}
