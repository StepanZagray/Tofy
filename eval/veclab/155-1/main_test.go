package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 2); math.Abs(got-5.3999999999999995) > 1e-9 { t.Fatalf("got %v want 5.3999999999999995", got) }
	if got := Solve("x1,y2", 2); math.Abs(got-1.7999999999999998) > 1e-9 { t.Fatalf("got %v want 1.7999999999999998", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
