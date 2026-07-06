package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1591 { t.Fatalf("Solve(-31) = %d, want 1591", got) }
	if got := Solve(-1); got != 2431 { t.Fatalf("Solve(-1) = %d, want 2431", got) }
	if got := Solve(0); got != 2451 { t.Fatalf("Solve(0) = %d, want 2451", got) }
	if got := Solve(7); got != 2655 { t.Fatalf("Solve(7) = %d, want 2655", got) }
	if got := Solve(91); got != 4975 { t.Fatalf("Solve(91) = %d, want 4975", got) }
}
