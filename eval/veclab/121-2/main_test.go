package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 3); math.Abs(got-176.39999999999998) > 1e-9 { t.Fatalf("got %v want 176.39999999999998", got) }
	if got := Solve("x1,y2", 3); math.Abs(got-268.79999999999995) > 1e-9 { t.Fatalf("got %v want 268.79999999999995", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
