package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -297 { t.Fatalf("Solve(-31) = %d, want -297", got) }
	if got := Solve(-1); got != 527 { t.Fatalf("Solve(-1) = %d, want 527", got) }
	if got := Solve(0); got != 555 { t.Fatalf("Solve(0) = %d, want 555", got) }
	if got := Solve(7); got != 751 { t.Fatalf("Solve(7) = %d, want 751", got) }
	if got := Solve(91); got != 3135 { t.Fatalf("Solve(91) = %d, want 3135", got) }
}
