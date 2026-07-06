package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -66 { t.Fatalf("Solve(-31) = %d, want -66", got) }
	if got := Solve(-1); got != 200 { t.Fatalf("Solve(-1) = %d, want 200", got) }
	if got := Solve(0); got != 195 { t.Fatalf("Solve(0) = %d, want 195", got) }
	if got := Solve(7); got != 224 { t.Fatalf("Solve(7) = %d, want 224", got) }
	if got := Solve(91); got != 540 { t.Fatalf("Solve(91) = %d, want 540", got) }
}
