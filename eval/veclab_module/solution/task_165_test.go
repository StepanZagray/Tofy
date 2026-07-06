package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1223 { t.Fatalf("Solve(-31) = %d, want 1223", got) }
	if got := Solve(-1); got != 1571 { t.Fatalf("Solve(-1) = %d, want 1571", got) }
	if got := Solve(0); got != 1589 { t.Fatalf("Solve(0) = %d, want 1589", got) }
	if got := Solve(7); got != 1971 { t.Fatalf("Solve(7) = %d, want 1971", got) }
	if got := Solve(91); got != 3499 { t.Fatalf("Solve(91) = %d, want 3499", got) }
}
