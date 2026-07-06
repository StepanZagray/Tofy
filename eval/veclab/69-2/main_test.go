package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-2.2)) > 1e-9 { t.Fatalf("got %v want -2.2", got) }
	if got := Solve([]float64{1, 2, 3}, 3); math.Abs(got-6.6000000000000005) > 1e-9 { t.Fatalf("got %v want 6.6000000000000005", got) }
	if got := Solve([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
