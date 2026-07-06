package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1330 { t.Fatalf("Solve(-31) = %d, want 1330", got) }
	if got := Solve(-1); got != 2084 { t.Fatalf("Solve(-1) = %d, want 2084", got) }
	if got := Solve(0); got != 2107 { t.Fatalf("Solve(0) = %d, want 2107", got) }
	if got := Solve(7); got != 2412 { t.Fatalf("Solve(7) = %d, want 2412", got) }
	if got := Solve(91); got != 4376 { t.Fatalf("Solve(91) = %d, want 4376", got) }
}
