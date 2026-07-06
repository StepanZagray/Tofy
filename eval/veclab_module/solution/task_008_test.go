package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -394 { t.Fatalf("Solve(-31) = %d, want -394", got) }
	if got := Solve(-1); got != 206 { t.Fatalf("Solve(-1) = %d, want 206", got) }
	if got := Solve(0); got != 226 { t.Fatalf("Solve(0) = %d, want 226", got) }
	if got := Solve(7); got != 366 { t.Fatalf("Solve(7) = %d, want 366", got) }
	if got := Solve(91); got != 2046 { t.Fatalf("Solve(91) = %d, want 2046", got) }
}
