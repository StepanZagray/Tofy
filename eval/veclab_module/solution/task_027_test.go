package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 172 { t.Fatalf("Solve(-31) = %d, want 172", got) }
	if got := Solve(-1); got != 468 { t.Fatalf("Solve(-1) = %d, want 468", got) }
	if got := Solve(0); got != 288 { t.Fatalf("Solve(0) = %d, want 288", got) }
	if got := Solve(7); got != 372 { t.Fatalf("Solve(7) = %d, want 372", got) }
	if got := Solve(91); got != 1380 { t.Fatalf("Solve(91) = %d, want 1380", got) }
}
