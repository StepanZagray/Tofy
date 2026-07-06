package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 93 { t.Fatalf("Solve(-31) = %d, want 93", got) }
	if got := Solve(-1); got != 467 { t.Fatalf("Solve(-1) = %d, want 467", got) }
	if got := Solve(0); got != 544 { t.Fatalf("Solve(0) = %d, want 544", got) }
	if got := Solve(7); got != 587 { t.Fatalf("Solve(7) = %d, want 587", got) }
	if got := Solve(91); got != 1679 { t.Fatalf("Solve(91) = %d, want 1679", got) }
}
