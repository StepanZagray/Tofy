package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 610 { t.Fatalf("Solve(-31) = %d, want 610", got) }
	if got := Solve(-1); got != 948 { t.Fatalf("Solve(-1) = %d, want 948", got) }
	if got := Solve(0); got != 953 { t.Fatalf("Solve(0) = %d, want 953", got) }
	if got := Solve(7); got != 972 { t.Fatalf("Solve(7) = %d, want 972", got) }
	if got := Solve(91); got != 1952 { t.Fatalf("Solve(91) = %d, want 1952", got) }
}
