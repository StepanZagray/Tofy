package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 3); math.Abs(got-251) > 1e-9 { t.Fatalf("got %v want 251", got) }
	if got := Solve("x1,y2", 3); math.Abs(got-383) > 1e-9 { t.Fatalf("got %v want 383", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
