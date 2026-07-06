package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 629 { t.Fatalf("Solve(-31) = %d, want 629", got) }
	if got := Solve(-1); got != 1287 { t.Fatalf("Solve(-1) = %d, want 1287", got) }
	if got := Solve(0); got != 1340 { t.Fatalf("Solve(0) = %d, want 1340", got) }
	if got := Solve(7); got != 1759 { t.Fatalf("Solve(7) = %d, want 1759", got) }
	if got := Solve(91); got != 3395 { t.Fatalf("Solve(91) = %d, want 3395", got) }
}
