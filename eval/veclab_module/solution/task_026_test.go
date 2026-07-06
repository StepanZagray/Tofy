package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1108 { t.Fatalf("Solve(-31) = %d, want 1108", got) }
	if got := Solve(-1); got != 1618 { t.Fatalf("Solve(-1) = %d, want 1618", got) }
	if got := Solve(0); got != 1605 { t.Fatalf("Solve(0) = %d, want 1605", got) }
	if got := Solve(7); got != 1738 { t.Fatalf("Solve(7) = %d, want 1738", got) }
	if got := Solve(91); got != 3198 { t.Fatalf("Solve(91) = %d, want 3198", got) }
}
