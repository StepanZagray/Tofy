package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -76 { t.Fatalf("Solve(-31) = %d, want -76", got) }
	if got := Solve(-1); got != 88 { t.Fatalf("Solve(-1) = %d, want 88", got) }
	if got := Solve(0); got != 66 { t.Fatalf("Solve(0) = %d, want 66", got) }
	if got := Solve(7); got != 8 { t.Fatalf("Solve(7) = %d, want 8", got) }
	if got := Solve(91); got != 1264 { t.Fatalf("Solve(91) = %d, want 1264", got) }
}
