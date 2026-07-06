package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 6); math.Abs(got-6.3) > 1e-9 { t.Fatalf("got %v want 6.3", got) }
	if got := Solve("x1,y2", 6); math.Abs(got-2.0999999999999996) > 1e-9 { t.Fatalf("got %v want 2.0999999999999996", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
