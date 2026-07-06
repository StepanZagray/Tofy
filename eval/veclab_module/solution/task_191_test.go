package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 219 { t.Fatalf("Solve(-31) = %d, want 219", got) }
	if got := Solve(-1); got != 339 { t.Fatalf("Solve(-1) = %d, want 339", got) }
	if got := Solve(0); got != 351 { t.Fatalf("Solve(0) = %d, want 351", got) }
	if got := Solve(7); got != 371 { t.Fatalf("Solve(7) = %d, want 371", got) }
	if got := Solve(91); got != 707 { t.Fatalf("Solve(91) = %d, want 707", got) }
}
