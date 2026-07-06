package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2, 4}, 5); math.Abs(got-2.7) > 1e-9 { t.Fatalf("got %v want 2.7", got) }
	if got := Solve([]float64{1, 2, 3}, 5); math.Abs(got-1.8) > 1e-9 { t.Fatalf("got %v want 1.8", got) }
	if got := Solve([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
