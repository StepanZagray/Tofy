package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 5); math.Abs(got-254) > 1e-9 { t.Fatalf("got %v want 254", got) }
	if got := Solve("x1,y2", 5); math.Abs(got-386) > 1e-9 { t.Fatalf("got %v want 386", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
