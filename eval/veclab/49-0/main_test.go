package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-16) > 1e-9 { t.Fatalf("got %v want 16", got) }
	if got := Solve([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Solve([]float64{0.5}, []float64{2}); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
}
