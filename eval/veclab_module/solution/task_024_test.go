package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -409 { t.Fatalf("Solve(-31) = %d, want -409", got) }
	if got := Solve(-1); got != 269 { t.Fatalf("Solve(-1) = %d, want 269", got) }
	if got := Solve(0); got != 288 { t.Fatalf("Solve(0) = %d, want 288", got) }
	if got := Solve(7); got != 613 { t.Fatalf("Solve(7) = %d, want 613", got) }
	if got := Solve(91); got != 3001 { t.Fatalf("Solve(91) = %d, want 3001", got) }
}
