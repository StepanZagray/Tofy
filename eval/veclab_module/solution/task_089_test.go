package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 1388 { t.Fatalf("Solve(-31) = %d, want 1388", got) }
	if got := Solve(-1); got != 1946 { t.Fatalf("Solve(-1) = %d, want 1946", got) }
	if got := Solve(0); got != 2023 { t.Fatalf("Solve(0) = %d, want 2023", got) }
	if got := Solve(7); got != 2162 { t.Fatalf("Solve(7) = %d, want 2162", got) }
	if got := Solve(91); got != 3950 { t.Fatalf("Solve(91) = %d, want 3950", got) }
}
