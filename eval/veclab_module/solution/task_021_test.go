package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -435 { t.Fatalf("Solve(-31) = %d, want -435", got) }
	if got := Solve(-1); got != 203 { t.Fatalf("Solve(-1) = %d, want 203", got) }
	if got := Solve(0); got != 216 { t.Fatalf("Solve(0) = %d, want 216", got) }
	if got := Solve(7); got != 99 { t.Fatalf("Solve(7) = %d, want 99", got) }
	if got := Solve(91); got != 1087 { t.Fatalf("Solve(91) = %d, want 1087", got) }
}
