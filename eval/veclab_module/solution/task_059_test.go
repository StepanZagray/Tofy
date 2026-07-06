package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 655 { t.Fatalf("Solve(-31) = %d, want 655", got) }
	if got := Solve(-1); got != 1411 { t.Fatalf("Solve(-1) = %d, want 1411", got) }
	if got := Solve(0); got != 1509 { t.Fatalf("Solve(0) = %d, want 1509", got) }
	if got := Solve(7); got != 1363 { t.Fatalf("Solve(7) = %d, want 1363", got) }
	if got := Solve(91); got != 3627 { t.Fatalf("Solve(91) = %d, want 3627", got) }
}
