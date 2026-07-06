package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 720 { t.Fatalf("Solve(-31) = %d, want 720", got) }
	if got := Solve(-1); got != 1076 { t.Fatalf("Solve(-1) = %d, want 1076", got) }
	if got := Solve(0); got != 1058 { t.Fatalf("Solve(0) = %d, want 1058", got) }
	if got := Solve(7); got != 1220 { t.Fatalf("Solve(7) = %d, want 1220", got) }
	if got := Solve(91); got != 2364 { t.Fatalf("Solve(91) = %d, want 2364", got) }
}
