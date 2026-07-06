package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 257 { t.Fatalf("Solve(-31) = %d, want 257", got) }
	if got := Solve(-1); got != 1087 { t.Fatalf("Solve(-1) = %d, want 1087", got) }
	if got := Solve(0); got != 1498 { t.Fatalf("Solve(0) = %d, want 1498", got) }
	if got := Solve(7); got != 1287 { t.Fatalf("Solve(7) = %d, want 1287", got) }
	if got := Solve(91); got != 3683 { t.Fatalf("Solve(91) = %d, want 3683", got) }
}
