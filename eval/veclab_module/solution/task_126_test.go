package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -123 { t.Fatalf("Solve(-31) = %d, want -123", got) }
	if got := Solve(-1); got != 241 { t.Fatalf("Solve(-1) = %d, want 241", got) }
	if got := Solve(0); got != 251 { t.Fatalf("Solve(0) = %d, want 251", got) }
	if got := Solve(7); got != 1 { t.Fatalf("Solve(7) = %d, want 1", got) }
	if got := Solve(91); got != 841 { t.Fatalf("Solve(91) = %d, want 841", got) }
}
