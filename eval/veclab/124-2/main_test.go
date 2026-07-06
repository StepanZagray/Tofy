package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 6); math.Abs(got-250) > 1e-9 { t.Fatalf("got %v want 250", got) }
	if got := Solve("x1,y2", 6); math.Abs(got-382) > 1e-9 { t.Fatalf("got %v want 382", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
