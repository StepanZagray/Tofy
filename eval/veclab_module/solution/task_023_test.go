package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 242 { t.Fatalf("Solve(-31) = %d, want 242", got) }
	if got := Solve(-1); got != 152 { t.Fatalf("Solve(-1) = %d, want 152", got) }
	if got := Solve(0); got != 149 { t.Fatalf("Solve(0) = %d, want 149", got) }
	if got := Solve(7); got != 128 { t.Fatalf("Solve(7) = %d, want 128", got) }
	if got := Solve(91); got != 388 { t.Fatalf("Solve(91) = %d, want 388", got) }
}
