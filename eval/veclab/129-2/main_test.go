package solution

import (
	"math"
	"testing"
)

func TestSolve(t *testing.T) {
	if got := Solve("ab9", 6); math.Abs(got-201.60000000000002) > 1e-9 { t.Fatalf("got %v want 201.60000000000002", got) }
	if got := Solve("x1,y2", 6); math.Abs(got-307.20000000000005) > 1e-9 { t.Fatalf("got %v want 307.20000000000005", got) }
	if got := Solve("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}
