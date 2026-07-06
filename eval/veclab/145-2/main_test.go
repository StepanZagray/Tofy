package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Solve("x1,y2", 2); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
