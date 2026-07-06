package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 514 { t.Fatalf("Solve(-31) = %d, want 514", got) }
	if got := Solve(-1); got != 1206 { t.Fatalf("Solve(-1) = %d, want 1206", got) }
	if got := Solve(0); got != 1192 { t.Fatalf("Solve(0) = %d, want 1192", got) }
	if got := Solve(7); got != 1350 { t.Fatalf("Solve(7) = %d, want 1350", got) }
	if got := Solve(91); got != 3230 { t.Fatalf("Solve(91) = %d, want 3230", got) }
}
