package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 2); math.Abs(got-277.20000000000005) > 1e-9 { t.Fatalf("got %v want 277.20000000000005", got) }
	if got := Solve("x1,y2", 2); math.Abs(got-422.40000000000003) > 1e-9 { t.Fatalf("got %v want 422.40000000000003", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
