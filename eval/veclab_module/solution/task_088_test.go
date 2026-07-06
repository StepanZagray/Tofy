package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 606 { t.Fatalf("Solve(-31) = %d, want 606", got) }
	if got := Solve(-1); got != 1322 { t.Fatalf("Solve(-1) = %d, want 1322", got) }
	if got := Solve(0); got != 1284 { t.Fatalf("Solve(0) = %d, want 1284", got) }
	if got := Solve(7); got != 1658 { t.Fatalf("Solve(7) = %d, want 1658", got) }
	if got := Solve(91); got != 4034 { t.Fatalf("Solve(91) = %d, want 4034", got) }
}
