package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 49 { t.Fatalf("Solve(-31) = %d, want 49", got) }
	if got := Solve(-1); got != 257 { t.Fatalf("Solve(-1) = %d, want 257", got) }
	if got := Solve(0); got != 265 { t.Fatalf("Solve(0) = %d, want 265", got) }
	if got := Solve(7); got != 705 { t.Fatalf("Solve(7) = %d, want 705", got) }
	if got := Solve(91); got != 1121 { t.Fatalf("Solve(91) = %d, want 1121", got) }
}
