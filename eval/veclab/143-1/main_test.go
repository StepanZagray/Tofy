package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 5); math.Abs(got-7.2) > 1e-9 { t.Fatalf("got %v want 7.2", got) }
	if got := Solve("x1,y2", 5); math.Abs(got-2.4000000000000004) > 1e-9 { t.Fatalf("got %v want 2.4000000000000004", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
