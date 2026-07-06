package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-1.6)) > 1e-9 { t.Fatalf("got %v want -1.6", got) }
	if got := Solve([]float64{1, 2, 3}, 3); math.Abs(got-4.800000000000001) > 1e-9 { t.Fatalf("got %v want 4.800000000000001", got) }
	if got := Solve([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
