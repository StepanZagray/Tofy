package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != -226 { t.Fatalf("Solve(-31) = %d, want -226", got) }
	if got := Solve(-1); got != 280 { t.Fatalf("Solve(-1) = %d, want 280", got) }
	if got := Solve(0); got != 361 { t.Fatalf("Solve(0) = %d, want 361", got) }
	if got := Solve(7); got != 400 { t.Fatalf("Solve(7) = %d, want 400", got) }
	if got := Solve(91); got != 1892 { t.Fatalf("Solve(91) = %d, want 1892", got) }
}
