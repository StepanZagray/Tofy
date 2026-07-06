package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 875 { t.Fatalf("Solve(-31) = %d, want 875", got) }
	if got := Solve(-1); got != 1661 { t.Fatalf("Solve(-1) = %d, want 1661", got) }
	if got := Solve(0); got != 1618 { t.Fatalf("Solve(0) = %d, want 1618", got) }
	if got := Solve(7); got != 1957 { t.Fatalf("Solve(7) = %d, want 1957", got) }
	if got := Solve(91); got != 3953 { t.Fatalf("Solve(91) = %d, want 3953", got) }
}
