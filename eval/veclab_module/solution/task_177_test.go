package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 369 { t.Fatalf("Solve(-31) = %d, want 369", got) }
	if got := Solve(-1); got != 849 { t.Fatalf("Solve(-1) = %d, want 849", got) }
	if got := Solve(0); got != 833 { t.Fatalf("Solve(0) = %d, want 833", got) }
	if got := Solve(7); got != 977 { t.Fatalf("Solve(7) = %d, want 977", got) }
	if got := Solve(91); got != 2321 { t.Fatalf("Solve(91) = %d, want 2321", got) }
}
