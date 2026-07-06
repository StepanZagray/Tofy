package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 5); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Solve("x1,y2", 5); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
