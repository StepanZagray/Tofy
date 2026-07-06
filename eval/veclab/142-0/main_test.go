package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 4); math.Abs(got-14) > 1e-9 { t.Fatalf("got %v want 14", got) }
	if got := Solve("x1,y2", 4); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
