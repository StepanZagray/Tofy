package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve([]float64{3, -7, 2}, 3); math.Abs(got-(-2.6666666666666665)) > 1e-9 { t.Fatalf("got %v want -2.6666666666666665", got) }
	if got := Solve([]float64{1}, 3); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Solve([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-0.3333333333333333)) > 1e-9 { t.Fatalf("got %v want -0.3333333333333333", got) }
}
