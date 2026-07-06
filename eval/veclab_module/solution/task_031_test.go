package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1157 { t.Fatalf("Solve(-31) = %d, want 1157", got) }
	if got := Solve(-1); got != 1659 { t.Fatalf("Solve(-1) = %d, want 1659", got) }
	if got := Solve(0); got != 1550 { t.Fatalf("Solve(0) = %d, want 1550", got) }
	if got := Solve(7); got != 1955 { t.Fatalf("Solve(7) = %d, want 1955", got) }
	if got := Solve(91); got != 3719 { t.Fatalf("Solve(91) = %d, want 3719", got) }
}
