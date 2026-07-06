package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -11 { t.Fatalf("Solve(-31) = %d, want -11", got) }
	if got := Solve(-1); got != 469 { t.Fatalf("Solve(-1) = %d, want 469", got) }
	if got := Solve(0); got != 453 { t.Fatalf("Solve(0) = %d, want 453", got) }
	if got := Solve(7); got != 341 { t.Fatalf("Solve(7) = %d, want 341", got) }
	if got := Solve(91); got != 1557 { t.Fatalf("Solve(91) = %d, want 1557", got) }
}
