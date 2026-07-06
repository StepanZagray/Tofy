package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 173 { t.Fatalf("Solve(-31) = %d, want 173", got) }
	if got := Solve(-1); got != 945 { t.Fatalf("Solve(-1) = %d, want 945", got) }
	if got := Solve(0); got != 1111 { t.Fatalf("Solve(0) = %d, want 1111", got) }
	if got := Solve(7); got != 1153 { t.Fatalf("Solve(7) = %d, want 1153", got) }
	if got := Solve(91); got != 3337 { t.Fatalf("Solve(91) = %d, want 3337", got) }
}
