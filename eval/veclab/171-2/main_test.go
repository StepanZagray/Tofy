package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}); math.Abs(got-13) > 1e-9 { t.Fatalf("got %v want 13", got) }
	if got := Solve([]float64{1}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Solve([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
