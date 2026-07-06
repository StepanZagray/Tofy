package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -92 { t.Fatalf("Solve(-31) = %d, want -92", got) }
	if got := Solve(-1); got != 208 { t.Fatalf("Solve(-1) = %d, want 208", got) }
	if got := Solve(0); got != 234 { t.Fatalf("Solve(0) = %d, want 234", got) }
	if got := Solve(7); got != 128 { t.Fatalf("Solve(7) = %d, want 128", got) }
	if got := Solve(91); got != 648 { t.Fatalf("Solve(91) = %d, want 648", got) }
}
