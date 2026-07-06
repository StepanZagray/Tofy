package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 508 { t.Fatalf("Solve(-31) = %d, want 508", got) }
	if got := Solve(-1); got != 1256 { t.Fatalf("Solve(-1) = %d, want 1256", got) }
	if got := Solve(0); got != 1238 { t.Fatalf("Solve(0) = %d, want 1238", got) }
	if got := Solve(7); got != 1304 { t.Fatalf("Solve(7) = %d, want 1304", got) }
	if got := Solve(91); got != 3472 { t.Fatalf("Solve(91) = %d, want 3472", got) }
}
