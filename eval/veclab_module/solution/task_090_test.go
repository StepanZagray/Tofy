package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 51 { t.Fatalf("Solve(-31) = %d, want 51", got) }
	if got := Solve(-1); got != 595 { t.Fatalf("Solve(-1) = %d, want 595", got) }
	if got := Solve(0); got != 579 { t.Fatalf("Solve(0) = %d, want 579", got) }
	if got := Solve(7); got != 723 { t.Fatalf("Solve(7) = %d, want 723", got) }
	if got := Solve(91); got != 1939 { t.Fatalf("Solve(91) = %d, want 1939", got) }
}
