package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -57 { t.Fatalf("Solve(-31) = %d, want -57", got) }
	if got := Solve(-1); got != 453 { t.Fatalf("Solve(-1) = %d, want 453", got) }
	if got := Solve(0); got != 500 { t.Fatalf("Solve(0) = %d, want 500", got) }
	if got := Solve(7); got != 637 { t.Fatalf("Solve(7) = %d, want 637", got) }
	if got := Solve(91); got != 2025 { t.Fatalf("Solve(91) = %d, want 2025", got) }
}
