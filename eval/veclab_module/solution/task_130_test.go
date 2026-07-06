package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 741 { t.Fatalf("Solve(-31) = %d, want 741", got) }
	if got := Solve(-1); got != 1187 { t.Fatalf("Solve(-1) = %d, want 1187", got) }
	if got := Solve(0); got != 1206 { t.Fatalf("Solve(0) = %d, want 1206", got) }
	if got := Solve(7); got != 1083 { t.Fatalf("Solve(7) = %d, want 1083", got) }
	if got := Solve(91); got != 3063 { t.Fatalf("Solve(91) = %d, want 3063", got) }
}
