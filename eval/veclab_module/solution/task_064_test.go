package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 370 { t.Fatalf("Solve(-31) = %d, want 370", got) }
	if got := Solve(-1); got != 1270 { t.Fatalf("Solve(-1) = %d, want 1270", got) }
	if got := Solve(0); got != 1296 { t.Fatalf("Solve(0) = %d, want 1296", got) }
	if got := Solve(7); got != 1478 { t.Fatalf("Solve(7) = %d, want 1478", got) }
	if got := Solve(91); got != 4014 { t.Fatalf("Solve(91) = %d, want 4014", got) }
}
