package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 3); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Solve("x1,y2", 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
