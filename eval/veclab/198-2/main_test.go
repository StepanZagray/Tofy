package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
	if got := Solve([]float64{1}, 3); math.Abs(got-0.2) > 1e-9 { t.Fatalf("got %v want 0.2", got) }
	if got := Solve([]float64{-2, 4, -1, 5}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
}
