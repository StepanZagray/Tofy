package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 287 { t.Fatalf("Solve(-31) = %d, want 287", got) }
	if got := Solve(-1); got != 527 { t.Fatalf("Solve(-1) = %d, want 527", got) }
	if got := Solve(0); got != 519 { t.Fatalf("Solve(0) = %d, want 519", got) }
	if got := Solve(7); got != 719 { t.Fatalf("Solve(7) = %d, want 719", got) }
	if got := Solve(91); got != 1327 { t.Fatalf("Solve(91) = %d, want 1327", got) }
}
