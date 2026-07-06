package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 721 { t.Fatalf("Solve(-31) = %d, want 721", got) }
	if got := Solve(-1); got != 907 { t.Fatalf("Solve(-1) = %d, want 907", got) }
	if got := Solve(0); got != 902 { t.Fatalf("Solve(0) = %d, want 902", got) }
	if got := Solve(7); got != 1139 { t.Fatalf("Solve(7) = %d, want 1139", got) }
	if got := Solve(91); got != 1951 { t.Fatalf("Solve(91) = %d, want 1951", got) }
}
