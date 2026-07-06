package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -139 { t.Fatalf("Solve(-31) = %d, want -139", got) }
	if got := Solve(-1); got != 177 { t.Fatalf("Solve(-1) = %d, want 177", got) }
	if got := Solve(0); got != 179 { t.Fatalf("Solve(0) = %d, want 179", got) }
	if got := Solve(7); got != 161 { t.Fatalf("Solve(7) = %d, want 161", got) }
	if got := Solve(91); got != 121 { t.Fatalf("Solve(91) = %d, want 121", got) }
}
