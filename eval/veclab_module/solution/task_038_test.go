package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 400 { t.Fatalf("Solve(-31) = %d, want 400", got) }
	if got := Solve(-1); got != 594 { t.Fatalf("Solve(-1) = %d, want 594", got) }
	if got := Solve(0); got != 577 { t.Fatalf("Solve(0) = %d, want 577", got) }
	if got := Solve(7); got != 986 { t.Fatalf("Solve(7) = %d, want 986", got) }
	if got := Solve(91); got != 2286 { t.Fatalf("Solve(91) = %d, want 2286", got) }
}
