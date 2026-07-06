package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -297 { t.Fatalf("Solve(-31) = %d, want -297", got) }
	if got := Solve(-1); got != 217 { t.Fatalf("Solve(-1) = %d, want 217", got) }
	if got := Solve(0); got != 200 { t.Fatalf("Solve(0) = %d, want 200", got) }
	if got := Solve(7); got != 81 { t.Fatalf("Solve(7) = %d, want 81", got) }
	if got := Solve(91); got != 1597 { t.Fatalf("Solve(91) = %d, want 1597", got) }
}
