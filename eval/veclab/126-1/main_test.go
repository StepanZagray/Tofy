package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 3); math.Abs(got-252) > 1e-9 { t.Fatalf("got %v want 252", got) }
	if got := Solve("x1,y2", 3); math.Abs(got-384) > 1e-9 { t.Fatalf("got %v want 384", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
