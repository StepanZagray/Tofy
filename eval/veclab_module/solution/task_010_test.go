package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 302 { t.Fatalf("Solve(-31) = %d, want 302", got) }
	if got := Solve(-1); got != 1018 { t.Fatalf("Solve(-1) = %d, want 1018", got) }
	if got := Solve(0); got != 996 { t.Fatalf("Solve(0) = %d, want 996", got) }
	if got := Solve(7); got != 938 { t.Fatalf("Solve(7) = %d, want 938", got) }
	if got := Solve(91); got != 1554 { t.Fatalf("Solve(91) = %d, want 1554", got) }
}
