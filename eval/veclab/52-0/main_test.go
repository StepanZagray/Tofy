package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-35) > 1e-9 { t.Fatalf("got %v want 35", got) }
	if got := Solve([]float64{2, -1}, []float64{3, 4}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Solve([]float64{0.5}, []float64{2}); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
}
