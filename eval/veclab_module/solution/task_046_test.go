package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 103 { t.Fatalf("Solve(-31) = %d, want 103", got) }
	if got := Solve(-1); got != 223 { t.Fatalf("Solve(-1) = %d, want 223", got) }
	if got := Solve(0); got != 227 { t.Fatalf("Solve(0) = %d, want 227", got) }
	if got := Solve(7); got != 255 { t.Fatalf("Solve(7) = %d, want 255", got) }
	if got := Solve(91); got != 591 { t.Fatalf("Solve(91) = %d, want 591", got) }
}
