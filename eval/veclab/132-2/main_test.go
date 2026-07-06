package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 4); math.Abs(got-247) > 1e-9 { t.Fatalf("got %v want 247", got) }
	if got := Solve("x1,y2", 4); math.Abs(got-379) > 1e-9 { t.Fatalf("got %v want 379", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
