package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -156 { t.Fatalf("Solve(-31) = %d, want -156", got) }
	if got := Solve(-1); got != 482 { t.Fatalf("Solve(-1) = %d, want 482", got) }
	if got := Solve(0); got != 465 { t.Fatalf("Solve(0) = %d, want 465", got) }
	if got := Solve(7); got != 378 { t.Fatalf("Solve(7) = %d, want 378", got) }
	if got := Solve(91); got != 1718 { t.Fatalf("Solve(91) = %d, want 1718", got) }
}
