package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 957 { t.Fatalf("Solve(-31) = %d, want 957", got) }
	if got := Solve(-1); got != 1865 { t.Fatalf("Solve(-1) = %d, want 1865", got) }
	if got := Solve(0); got != 1879 { t.Fatalf("Solve(0) = %d, want 1879", got) }
	if got := Solve(7); got != 1945 { t.Fatalf("Solve(7) = %d, want 1945", got) }
	if got := Solve(91); got != 4113 { t.Fatalf("Solve(91) = %d, want 4113", got) }
}
