package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}); math.Abs(got-(-0.4666666666666666)) > 1e-9 { t.Fatalf("got %v want -0.4666666666666666", got) }
	if got := Solve([]float64{1}); math.Abs(got-0.7) > 1e-9 { t.Fatalf("got %v want 0.7", got) }
	if got := Solve([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
