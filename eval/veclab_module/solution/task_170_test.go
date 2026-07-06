package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1202 { t.Fatalf("Solve(-31) = %d, want 1202", got) }
	if got := Solve(-1); got != 1838 { t.Fatalf("Solve(-1) = %d, want 1838", got) }
	if got := Solve(0); got != 1880 { t.Fatalf("Solve(0) = %d, want 1880", got) }
	if got := Solve(7); got != 2046 { t.Fatalf("Solve(7) = %d, want 2046", got) }
	if got := Solve(91); got != 3894 { t.Fatalf("Solve(91) = %d, want 3894", got) }
}
