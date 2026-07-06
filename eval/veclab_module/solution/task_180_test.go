package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1246 { t.Fatalf("Solve(-31) = %d, want 1246", got) }
	if got := Solve(-1); got != 1882 { t.Fatalf("Solve(-1) = %d, want 1882", got) }
	if got := Solve(0); got != 1852 { t.Fatalf("Solve(0) = %d, want 1852", got) }
	if got := Solve(7); got != 2122 { t.Fatalf("Solve(7) = %d, want 2122", got) }
	if got := Solve(91); got != 4706 { t.Fatalf("Solve(91) = %d, want 4706", got) }
}
