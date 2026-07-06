package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 48 { t.Fatalf("Solve(-31) = %d, want 48", got) }
	if got := Solve(-1); got != 440 { t.Fatalf("Solve(-1) = %d, want 440", got) }
	if got := Solve(0); got != 436 { t.Fatalf("Solve(0) = %d, want 436", got) }
	if got := Solve(7); got != 472 { t.Fatalf("Solve(7) = %d, want 472", got) }
	if got := Solve(91); got != 520 { t.Fatalf("Solve(91) = %d, want 520", got) }
}
