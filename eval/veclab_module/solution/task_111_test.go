package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 227 { t.Fatalf("Solve(-31) = %d, want 227", got) }
	if got := Solve(-1); got != 981 { t.Fatalf("Solve(-1) = %d, want 981", got) }
	if got := Solve(0); got != 936 { t.Fatalf("Solve(0) = %d, want 936", got) }
	if got := Solve(7); got != 1261 { t.Fatalf("Solve(7) = %d, want 1261", got) }
	if got := Solve(91); got != 3329 { t.Fatalf("Solve(91) = %d, want 3329", got) }
}
