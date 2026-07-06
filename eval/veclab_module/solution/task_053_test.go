package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -70 { t.Fatalf("Solve(-31) = %d, want -70", got) }
	if got := Solve(-1); got != 660 { t.Fatalf("Solve(-1) = %d, want 660", got) }
	if got := Solve(0); got != 625 { t.Fatalf("Solve(0) = %d, want 625", got) }
	if got := Solve(7); got != 956 { t.Fatalf("Solve(7) = %d, want 956", got) }
	if got := Solve(91); got != 3288 { t.Fatalf("Solve(91) = %d, want 3288", got) }
}
