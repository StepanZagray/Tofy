package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 363 { t.Fatalf("Solve(-31) = %d, want 363", got) }
	if got := Solve(-1); got != 733 { t.Fatalf("Solve(-1) = %d, want 733", got) }
	if got := Solve(0); got != 704 { t.Fatalf("Solve(0) = %d, want 704", got) }
	if got := Solve(7); got != 693 { t.Fatalf("Solve(7) = %d, want 693", got) }
	if got := Solve(91); got != 1577 { t.Fatalf("Solve(91) = %d, want 1577", got) }
}
