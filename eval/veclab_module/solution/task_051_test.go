package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 235 { t.Fatalf("Solve(-31) = %d, want 235", got) }
	if got := Solve(-1); got != 283 { t.Fatalf("Solve(-1) = %d, want 283", got) }
	if got := Solve(0); got != 275 { t.Fatalf("Solve(0) = %d, want 275", got) }
	if got := Solve(7); got != 475 { t.Fatalf("Solve(7) = %d, want 475", got) }
	if got := Solve(91); got != 1083 { t.Fatalf("Solve(91) = %d, want 1083", got) }
}
