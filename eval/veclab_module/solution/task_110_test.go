package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 222 { t.Fatalf("Solve(-31) = %d, want 222", got) }
	if got := Solve(-1); got != 256 { t.Fatalf("Solve(-1) = %d, want 256", got) }
	if got := Solve(0); got != 317 { t.Fatalf("Solve(0) = %d, want 317", got) }
	if got := Solve(7); got != 296 { t.Fatalf("Solve(7) = %d, want 296", got) }
	if got := Solve(91); got != 556 { t.Fatalf("Solve(91) = %d, want 556", got) }
}
