package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -378 { t.Fatalf("Solve(-31) = %d, want -378", got) }
	if got := Solve(-1); got != 122 { t.Fatalf("Solve(-1) = %d, want 122", got) }
	if got := Solve(0); got != 112 { t.Fatalf("Solve(0) = %d, want 112", got) }
	if got := Solve(7); got != 10 { t.Fatalf("Solve(7) = %d, want 10", got) }
	if got := Solve(91); got != 962 { t.Fatalf("Solve(91) = %d, want 962", got) }
}
