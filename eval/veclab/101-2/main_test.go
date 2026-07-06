package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}, 3); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Solve([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Solve([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}
