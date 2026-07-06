package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 91 { t.Fatalf("Solve(-31) = %d, want 91", got) }
	if got := Solve(-1); got != 541 { t.Fatalf("Solve(-1) = %d, want 541", got) }
	if got := Solve(0); got != 554 { t.Fatalf("Solve(0) = %d, want 554", got) }
	if got := Solve(7); got != 901 { t.Fatalf("Solve(7) = %d, want 901", got) }
	if got := Solve(91); got != 1905 { t.Fatalf("Solve(91) = %d, want 1905", got) }
}
