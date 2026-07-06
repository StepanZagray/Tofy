package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -44 { t.Fatalf("Solve(-31) = %d, want -44", got) }
	if got := Solve(-1); got != 830 { t.Fatalf("Solve(-1) = %d, want 830", got) }
	if got := Solve(0); got != 795 { t.Fatalf("Solve(0) = %d, want 795", got) }
	if got := Solve(7); got != 982 { t.Fatalf("Solve(7) = %d, want 982", got) }
	if got := Solve(91); got != 3314 { t.Fatalf("Solve(91) = %d, want 3314", got) }
}
