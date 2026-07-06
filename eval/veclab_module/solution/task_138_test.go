package solution

import "testing"

func TestSolve(t *testing.T) {
	if got := Solve(-31); got != 281 { t.Fatalf("Solve(-31) = %d, want 281", got) }
	if got := Solve(-1); got != 521 { t.Fatalf("Solve(-1) = %d, want 521", got) }
	if got := Solve(0); got != 513 { t.Fatalf("Solve(0) = %d, want 513", got) }
	if got := Solve(7); got != 585 { t.Fatalf("Solve(7) = %d, want 585", got) }
	if got := Solve(91); got != 1257 { t.Fatalf("Solve(91) = %d, want 1257", got) }
}
