package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 4); math.Abs(got-151.2) > 1e-9 { t.Fatalf("got %v want 151.2", got) }
	if got := Solve("x1,y2", 4); math.Abs(got-230.39999999999998) > 1e-9 { t.Fatalf("got %v want 230.39999999999998", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
