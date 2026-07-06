package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 129 { t.Fatalf("Solve(-31) = %d, want 129", got) }
	if got := Solve(-1); got != 977 { t.Fatalf("Solve(-1) = %d, want 977", got) }
	if got := Solve(0); got != 937 { t.Fatalf("Solve(0) = %d, want 937", got) }
	if got := Solve(7); got != 785 { t.Fatalf("Solve(7) = %d, want 785", got) }
	if got := Solve(91); got != 2865 { t.Fatalf("Solve(91) = %d, want 2865", got) }
}
