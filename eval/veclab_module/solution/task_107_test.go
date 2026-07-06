package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 981 { t.Fatalf("Solve(-31) = %d, want 981", got) }
	if got := Solve(-1); got != 1521 { t.Fatalf("Solve(-1) = %d, want 1521", got) }
	if got := Solve(0); got != 1415 { t.Fatalf("Solve(0) = %d, want 1415", got) }
	if got := Solve(7); got != 1537 { t.Fatalf("Solve(7) = %d, want 1537", got) }
	if got := Solve(91); got != 3177 { t.Fatalf("Solve(91) = %d, want 3177", got) }
}
