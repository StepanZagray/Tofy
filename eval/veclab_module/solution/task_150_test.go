package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -70 { t.Fatalf("Solve(-31) = %d, want -70", got) }
	if got := Solve(-1); got != 230 { t.Fatalf("Solve(-1) = %d, want 230", got) }
	if got := Solve(0); got != 252 { t.Fatalf("Solve(0) = %d, want 252", got) }
	if got := Solve(7); got != 214 { t.Fatalf("Solve(7) = %d, want 214", got) }
	if got := Solve(91); got != 734 { t.Fatalf("Solve(91) = %d, want 734", got) }
}
