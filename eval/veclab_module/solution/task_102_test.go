package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 926 { t.Fatalf("Solve(-31) = %d, want 926", got) }
	if got := Solve(-1); got != 1082 { t.Fatalf("Solve(-1) = %d, want 1082", got) }
	if got := Solve(0); got != 1068 { t.Fatalf("Solve(0) = %d, want 1068", got) }
	if got := Solve(7); got != 1418 { t.Fatalf("Solve(7) = %d, want 1418", got) }
	if got := Solve(91); got != 2338 { t.Fatalf("Solve(91) = %d, want 2338", got) }
}
