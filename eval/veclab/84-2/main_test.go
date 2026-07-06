package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2, 4}, 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Solve([]float64{1, 2, 3}, 2); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Solve([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
