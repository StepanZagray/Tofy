package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}, 4); math.Abs(got-(-0.5)) > 1e-9 { t.Fatalf("got %v want -0.5", got) }
	if got := Solve([]float64{1}, 3); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
	if got := Solve([]float64{-2, 4, -1, 5}, 4); math.Abs(got-(-1.5)) > 1e-9 { t.Fatalf("got %v want -1.5", got) }
}
