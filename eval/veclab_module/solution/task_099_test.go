package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1335 { t.Fatalf("Solve(-31) = %d, want 1335", got) }
	if got := Solve(-1); got != 2205 { t.Fatalf("Solve(-1) = %d, want 2205", got) }
	if got := Solve(0); got != 2232 { t.Fatalf("Solve(0) = %d, want 2232", got) }
	if got := Solve(7); got != 2437 { t.Fatalf("Solve(7) = %d, want 2437", got) }
	if got := Solve(91); got != 4873 { t.Fatalf("Solve(91) = %d, want 4873", got) }
}
