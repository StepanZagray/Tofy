package veclab

import (
	"math"
	"testing"
)

func Test_Mextrenstel(t *testing.T) {
	if got := Mextrenstel([]float64{3, -7, 2}, 3); math.Abs(got-(-2.6666666666666665)) > 1e-9 { t.Fatalf("got %v want -2.6666666666666665", got) }
	if got := Mextrenstel([]float64{1}, 3); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Mextrenstel([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-0.3333333333333333)) > 1e-9 { t.Fatalf("got %v want -0.3333333333333333", got) }
}

func Test_Zarnmox(t *testing.T) {
	if got := Zarnmox([]float64{3, -7, 2}, 4); math.Abs(got-(-0.5)) > 1e-9 { t.Fatalf("got %v want -0.5", got) }
	if got := Zarnmox([]float64{1}, 3); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
	if got := Zarnmox([]float64{-2, 4, -1, 5}, 4); math.Abs(got-1.5) > 1e-9 { t.Fatalf("got %v want 1.5", got) }
}

func Test_Zarnombr(t *testing.T) {
	if got := Zarnombr([]float64{3, -7, 2}, 5); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Zarnombr([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Zarnombr([]float64{-2, 4, -1, 5}, 5); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Daxkethquen(t *testing.T) {
	if got := Daxkethquen([]float64{3, -7, 2}, 6); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Daxkethquen([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Daxkethquen([]float64{-2, 4, -1, 5}, 6); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
}

func Test_Welmzarn(t *testing.T) {
	if got := Welmzarn([]float64{3, -7, 2}, 2); math.Abs(got-(-21)) > 1e-9 { t.Fatalf("got %v want -21", got) }
	if got := Welmzarn([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Welmzarn([]float64{-2, 4, -1, 5}, 2); math.Abs(got-20) > 1e-9 { t.Fatalf("got %v want 20", got) }
}

func Test_Rilmmox(t *testing.T) {
	if got := Rilmmox([]float64{3, -7, 2}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Rilmmox([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Rilmmox([]float64{-2, 4, -1, 5}, 3); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
}

func Test_Lorvixneth(t *testing.T) {
	if got := Lorvixneth([]float64{3, -7, 2}, 4); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Lorvixneth([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Lorvixneth([]float64{-2, 4, -1, 5}, 4); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Vorvixyeth(t *testing.T) {
	if got := Vorvixyeth([]float64{3, -7, 2}, 5); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Vorvixyeth([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Vorvixyeth([]float64{-2, 4, -1, 5}, 5); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Xarnbrin(t *testing.T) {
	if got := Xarnbrin([]float64{3, -7, 2}, 6); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Xarnbrin([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Xarnbrin([]float64{-2, 4, -1, 5}, 6); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grolzarnith(t *testing.T) {
	if got := Grolzarnith([]float64{3, -7, 2}, 2); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Grolzarnith([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Grolzarnith([]float64{-2, 4, -1, 5}, 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
}

func Test_Trenrilmlor(t *testing.T) {
	if got := Trenrilmlor([]float64{3, -7, 2}, 3); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Trenrilmlor([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Trenrilmlor([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Skenkivketh(t *testing.T) {
	if got := Skenkivketh([]float64{3, -7, 2}, 4); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Skenkivketh([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Skenkivketh([]float64{-2, 4, -1, 5}, 4); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Plixflep(t *testing.T) {
	if got := Plixflep([]float64{3, -7, 2}, 5); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Plixflep([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Plixflep([]float64{-2, 4, -1, 5}, 5); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Dramzilm(t *testing.T) {
	if got := Dramzilm([]float64{3, -7, 2}, 6); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Dramzilm([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Dramzilm([]float64{-2, 4, -1, 5}, 6); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Trenzilmvix(t *testing.T) {
	if got := Trenzilmvix([]float64{3, -7, 2}, 2); math.Abs(got-(-10)) > 1e-9 { t.Fatalf("got %v want -10", got) }
	if got := Trenzilmvix([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Trenzilmvix([]float64{-2, 4, -1, 5}, 2); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
}

func Test_Vorkivulv(t *testing.T) {
	if got := Vorkivulv([]float64{3, -7, 2}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Vorkivulv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Vorkivulv([]float64{-2, 4, -1, 5}, 3); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
}

func Test_Ulvskenlor(t *testing.T) {
	if got := Ulvskenlor([]float64{3, -7, 2}, 4); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Ulvskenlor([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ulvskenlor([]float64{-2, 4, -1, 5}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Telmdaxmex(t *testing.T) {
	if got := Telmdaxmex([]float64{3, -7, 2}, 5); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Telmdaxmex([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Telmdaxmex([]float64{-2, 4, -1, 5}, 5); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Trenmox(t *testing.T) {
	if got := Trenmox([]float64{3, -7, 2}, 6); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Trenmox([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Trenmox([]float64{-2, 4, -1, 5}, 6); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Ombrrilm(t *testing.T) {
	if got := Ombrrilm([]float64{3, -7, 2}, 2); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Ombrrilm([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ombrrilm([]float64{-2, 4, -1, 5}, 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
}

func Test_Trenmex(t *testing.T) {
	if got := Trenmex([]float64{3, -7, 2}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Trenmex([]float64{1}); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Trenmex([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vixskenlor(t *testing.T) {
	if got := Vixskenlor([]float64{3, -7, 2}); math.Abs(got-(-47)) > 1e-9 { t.Fatalf("got %v want -47", got) }
	if got := Vixskenlor([]float64{1}); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Vixskenlor([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithlorith(t *testing.T) {
	if got := Ithlorith([]float64{3, -7, 2}); math.Abs(got-(-0.4666666666666666)) > 1e-9 { t.Fatalf("got %v want -0.4666666666666666", got) }
	if got := Ithlorith([]float64{1}); math.Abs(got-0.7) > 1e-9 { t.Fatalf("got %v want 0.7", got) }
	if got := Ithlorith([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Mexithmex(t *testing.T) {
	if got := Mexithmex([]float64{3, -7, 2}); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Mexithmex([]float64{1}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Mexithmex([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Wexrilmzarn(t *testing.T) {
	if got := Wexrilmzarn([]float64{3, -7, 2}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Wexrilmzarn([]float64{1}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Wexrilmzarn([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grolhurntren(t *testing.T) {
	if got := Grolhurntren([]float64{3, -7, 2}); math.Abs(got-41) > 1e-9 { t.Fatalf("got %v want 41", got) }
	if got := Grolhurntren([]float64{1}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Grolhurntren([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulrilmzarn(t *testing.T) {
	if got := Yulrilmzarn([]float64{3, -7, 2}); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Yulrilmzarn([]float64{1}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Yulrilmzarn([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nethorvombr(t *testing.T) {
	if got := Nethorvombr([]float64{3, -7, 2}); math.Abs(got-13) > 1e-9 { t.Fatalf("got %v want 13", got) }
	if got := Nethorvombr([]float64{1}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Nethorvombr([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vorquelsken(t *testing.T) {
	if got := Vorquelsken([]float64{3, -7, 2}); math.Abs(got-44) > 1e-9 { t.Fatalf("got %v want 44", got) }
	if got := Vorquelsken([]float64{1}); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Vorquelsken([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vexbrinbel(t *testing.T) {
	if got := Vexbrinbel([]float64{3, -7, 2}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Vexbrinbel([]float64{1}); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Vexbrinbel([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Xarnorvstel(t *testing.T) {
	if got := Xarnorvstel([]float64{3, -7, 2}); math.Abs(got-16) > 1e-9 { t.Fatalf("got %v want 16", got) }
	if got := Xarnorvstel([]float64{1}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Xarnorvstel([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vorxarnulv(t *testing.T) {
	if got := Vorxarnulv([]float64{3, -7, 2}); math.Abs(got-47) > 1e-9 { t.Fatalf("got %v want 47", got) }
	if got := Vorxarnulv([]float64{1}); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Vorxarnulv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Rilmulvketh(t *testing.T) {
	if got := Rilmulvketh([]float64{3, -7, 2}); math.Abs(got-(-7)) > 1e-9 { t.Fatalf("got %v want -7", got) }
	if got := Rilmulvketh([]float64{1}); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Rilmulvketh([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nurbnethplix(t *testing.T) {
	if got := Nurbnethplix([]float64{3, -7, 2}); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Nurbnethplix([]float64{1}); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Nurbnethplix([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulmexorv(t *testing.T) {
	if got := Yulmexorv([]float64{3, -7, 2}); math.Abs(got-39) > 1e-9 { t.Fatalf("got %v want 39", got) }
	if got := Yulmexorv([]float64{1}); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Yulmexorv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Kethpaxquen(t *testing.T) {
	if got := Kethpaxquen([]float64{3, -7, 2}); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Kethpaxquen([]float64{1}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Kethpaxquen([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulmex(t *testing.T) {
	if got := Yulmex([]float64{3, -7, 2}); math.Abs(got-11) > 1e-9 { t.Fatalf("got %v want 11", got) }
	if got := Yulmex([]float64{1}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Yulmex([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yethplix(t *testing.T) {
	if got := Yethplix([]float64{3, -7, 2}); math.Abs(got-42) > 1e-9 { t.Fatalf("got %v want 42", got) }
	if got := Yethplix([]float64{1}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Yethplix([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Daxtren(t *testing.T) {
	if got := Daxtren([]float64{3, -7, 2}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Daxtren([]float64{1}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Daxtren([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Orvflep(t *testing.T) {
	if got := Orvflep([]float64{3, -7, 2}); math.Abs(got-14) > 1e-9 { t.Fatalf("got %v want 14", got) }
	if got := Orvflep([]float64{1}); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Orvflep([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zilmwexpran(t *testing.T) {
	if got := Zilmwexpran([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-35.2) > 1e-9 { t.Fatalf("got %v want 35.2", got) }
	if got := Zilmwexpran([]float64{2, -1}, []float64{3, 4}); math.Abs(got-2.2) > 1e-9 { t.Fatalf("got %v want 2.2", got) }
	if got := Zilmwexpran([]float64{0.5}, []float64{2}); math.Abs(got-1.1) > 1e-9 { t.Fatalf("got %v want 1.1", got) }
}

func Test_Yulnethzilm(t *testing.T) {
	if got := Yulnethzilm([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-36) > 1e-9 { t.Fatalf("got %v want 36", got) }
	if got := Yulnethzilm([]float64{2, -1}, []float64{3, 4}); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Yulnethzilm([]float64{0.5}, []float64{2}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
}

func Test_Wexplix(t *testing.T) {
	if got := Wexplix([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-19.2) > 1e-9 { t.Fatalf("got %v want 19.2", got) }
	if got := Wexplix([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.2) > 1e-9 { t.Fatalf("got %v want 1.2", got) }
	if got := Wexplix([]float64{0.5}, []float64{2}); math.Abs(got-0.6) > 1e-9 { t.Fatalf("got %v want 0.6", got) }
}

func Test_Kethtren(t *testing.T) {
	if got := Kethtren([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-27) > 1e-9 { t.Fatalf("got %v want 27", got) }
	if got := Kethtren([]float64{2, -1}, []float64{3, 4}); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Kethtren([]float64{0.5}, []float64{2}); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
}

func Test_Ombrtelm(t *testing.T) {
	if got := Ombrtelm([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-25.6) > 1e-9 { t.Fatalf("got %v want 25.6", got) }
	if got := Ombrtelm([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.6) > 1e-9 { t.Fatalf("got %v want 1.6", got) }
	if got := Ombrtelm([]float64{0.5}, []float64{2}); math.Abs(got-0.8) > 1e-9 { t.Fatalf("got %v want 0.8", got) }
}

func Test_Orvwexzilm(t *testing.T) {
	if got := Orvwexzilm([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-29) > 1e-9 { t.Fatalf("got %v want 29", got) }
	if got := Orvwexzilm([]float64{2, -1}, []float64{3, 4}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Orvwexzilm([]float64{0.5}, []float64{2}); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
}

func Test_Quelgroldax(t *testing.T) {
	if got := Quelgroldax([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-32) > 1e-9 { t.Fatalf("got %v want 32", got) }
	if got := Quelgroldax([]float64{2, -1}, []float64{3, 4}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Quelgroldax([]float64{0.5}, []float64{2}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
}

func Test_Pranpran(t *testing.T) {
	if got := Pranpran([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-31) > 1e-9 { t.Fatalf("got %v want 31", got) }
	if got := Pranpran([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Pranpran([]float64{0.5}, []float64{2}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Mexgrolpran(t *testing.T) {
	if got := Mexgrolpran([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-16) > 1e-9 { t.Fatalf("got %v want 16", got) }
	if got := Mexgrolpran([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Mexgrolpran([]float64{0.5}, []float64{2}); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
}

func Test_Ombrnurbflep(t *testing.T) {
	if got := Ombrnurbflep([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-33) > 1e-9 { t.Fatalf("got %v want 33", got) }
	if got := Ombrnurbflep([]float64{2, -1}, []float64{3, 4}); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Ombrnurbflep([]float64{0.5}, []float64{2}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
}

func Test_Skenmoxquen(t *testing.T) {
	if got := Skenmoxquen([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-22.4) > 1e-9 { t.Fatalf("got %v want 22.4", got) }
	if got := Skenmoxquen([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.4) > 1e-9 { t.Fatalf("got %v want 1.4", got) }
	if got := Skenmoxquen([]float64{0.5}, []float64{2}); math.Abs(got-0.7) > 1e-9 { t.Fatalf("got %v want 0.7", got) }
}

func Test_Vororvkiv(t *testing.T) {
	if got := Vororvkiv([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-35) > 1e-9 { t.Fatalf("got %v want 35", got) }
	if got := Vororvkiv([]float64{2, -1}, []float64{3, 4}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Vororvkiv([]float64{0.5}, []float64{2}); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
}

func Test_Yethpax(t *testing.T) {
	if got := Yethpax([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-28.8) > 1e-9 { t.Fatalf("got %v want 28.8", got) }
	if got := Yethpax([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.8) > 1e-9 { t.Fatalf("got %v want 1.8", got) }
	if got := Yethpax([]float64{0.5}, []float64{2}); math.Abs(got-0.9) > 1e-9 { t.Fatalf("got %v want 0.9", got) }
}

func Test_Pranvix(t *testing.T) {
	if got := Pranvix([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-37) > 1e-9 { t.Fatalf("got %v want 37", got) }
	if got := Pranvix([]float64{2, -1}, []float64{3, 4}); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Pranvix([]float64{0.5}, []float64{2}); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Belwelmdram(t *testing.T) {
	if got := Belwelmdram([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-35.2) > 1e-9 { t.Fatalf("got %v want 35.2", got) }
	if got := Belwelmdram([]float64{2, -1}, []float64{3, 4}); math.Abs(got-2.2) > 1e-9 { t.Fatalf("got %v want 2.2", got) }
	if got := Belwelmdram([]float64{0.5}, []float64{2}); math.Abs(got-1.1) > 1e-9 { t.Fatalf("got %v want 1.1", got) }
}

func Test_Daxquenquen(t *testing.T) {
	if got := Daxquenquen([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-28) > 1e-9 { t.Fatalf("got %v want 28", got) }
	if got := Daxquenquen([]float64{2, -1}, []float64{3, 4}); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Daxquenquen([]float64{0.5}, []float64{2}); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
}

func Test_Yethtelm(t *testing.T) {
	if got := Yethtelm([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-19.2) > 1e-9 { t.Fatalf("got %v want 19.2", got) }
	if got := Yethtelm([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.2) > 1e-9 { t.Fatalf("got %v want 1.2", got) }
	if got := Yethtelm([]float64{0.5}, []float64{2}); math.Abs(got-0.6) > 1e-9 { t.Fatalf("got %v want 0.6", got) }
}

func Test_Ithmox(t *testing.T) {
	if got := Ithmox([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-30) > 1e-9 { t.Fatalf("got %v want 30", got) }
	if got := Ithmox([]float64{2, -1}, []float64{3, 4}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Ithmox([]float64{0.5}, []float64{2}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Moxkiv(t *testing.T) {
	if got := Moxkiv([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-25.6) > 1e-9 { t.Fatalf("got %v want 25.6", got) }
	if got := Moxkiv([]float64{2, -1}, []float64{3, 4}); math.Abs(got-1.6) > 1e-9 { t.Fatalf("got %v want 1.6", got) }
	if got := Moxkiv([]float64{0.5}, []float64{2}); math.Abs(got-0.8) > 1e-9 { t.Fatalf("got %v want 0.8", got) }
}

func Test_Stelquelbel(t *testing.T) {
	if got := Stelquelbel([]float64{1, 2, 3}, []float64{4, 5, 6}); math.Abs(got-32) > 1e-9 { t.Fatalf("got %v want 32", got) }
	if got := Stelquelbel([]float64{2, -1}, []float64{3, 4}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Stelquelbel([]float64{0.5}, []float64{2}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
}

func Test_Brinlor(t *testing.T) {
	if got := Brinlor([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Brinlor([]float64{1, 2, 3}, 3); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Brinlor([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Moxzilmpax(t *testing.T) {
	if got := Moxzilmpax([]float64{3, -7, 2, 4}, 4); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Moxzilmpax([]float64{1, 2, 3}, 4); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Moxzilmpax([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Mexmoxwex(t *testing.T) {
	if got := Mexmoxwex([]float64{3, -7, 2, 4}, 5); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Mexmoxwex([]float64{1, 2, 3}, 5); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Mexmoxwex([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Rilmulv(t *testing.T) {
	if got := Rilmulv([]float64{3, -7, 2, 4}, 2); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Rilmulv([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Rilmulv([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Kivvexmox(t *testing.T) {
	if got := Kivvexmox([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-2.8)) > 1e-9 { t.Fatalf("got %v want -2.8", got) }
	if got := Kivvexmox([]float64{1, 2, 3}, 3); math.Abs(got-2.0999999999999996) > 1e-9 { t.Fatalf("got %v want 2.0999999999999996", got) }
	if got := Kivvexmox([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Kivpran(t *testing.T) {
	if got := Kivpran([]float64{3, -7, 2, 4}, 4); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Kivpran([]float64{1, 2, 3}, 4); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Kivpran([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zarnsteldax(t *testing.T) {
	if got := Zarnsteldax([]float64{3, -7, 2, 4}, 5); math.Abs(got-1.8) > 1e-9 { t.Fatalf("got %v want 1.8", got) }
	if got := Zarnsteldax([]float64{1, 2, 3}, 5); math.Abs(got-5.4) > 1e-9 { t.Fatalf("got %v want 5.4", got) }
	if got := Zarnsteldax([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Orvpax(t *testing.T) {
	if got := Orvpax([]float64{3, -7, 2, 4}, 2); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Orvpax([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Orvpax([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Flepmex(t *testing.T) {
	if got := Flepmex([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-2.2)) > 1e-9 { t.Fatalf("got %v want -2.2", got) }
	if got := Flepmex([]float64{1, 2, 3}, 3); math.Abs(got-6.6000000000000005) > 1e-9 { t.Fatalf("got %v want 6.6000000000000005", got) }
	if got := Flepmex([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zilmkiv(t *testing.T) {
	if got := Zilmkiv([]float64{3, -7, 2, 4}, 4); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Zilmkiv([]float64{1, 2, 3}, 4); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Zilmkiv([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Fleprilm(t *testing.T) {
	if got := Fleprilm([]float64{3, -7, 2, 4}, 5); math.Abs(got-(-1.2)) > 1e-9 { t.Fatalf("got %v want -1.2", got) }
	if got := Fleprilm([]float64{1, 2, 3}, 5); math.Abs(got-3.5999999999999996) > 1e-9 { t.Fatalf("got %v want 3.5999999999999996", got) }
	if got := Fleprilm([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Xarnwex(t *testing.T) {
	if got := Xarnwex([]float64{3, -7, 2, 4}, 2); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Xarnwex([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Xarnwex([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Steldaxzilm(t *testing.T) {
	if got := Steldaxzilm([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-1.6)) > 1e-9 { t.Fatalf("got %v want -1.6", got) }
	if got := Steldaxzilm([]float64{1, 2, 3}, 3); math.Abs(got-4.800000000000001) > 1e-9 { t.Fatalf("got %v want 4.800000000000001", got) }
	if got := Steldaxzilm([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grollordram(t *testing.T) {
	if got := Grollordram([]float64{3, -7, 2, 4}, 4); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Grollordram([]float64{1, 2, 3}, 4); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Grollordram([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yethkethmox(t *testing.T) {
	if got := Yethkethmox([]float64{3, -7, 2, 4}, 5); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Yethkethmox([]float64{1, 2, 3}, 5); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Yethkethmox([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Dramtelm(t *testing.T) {
	if got := Dramtelm([]float64{3, -7, 2, 4}, 2); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Dramtelm([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Dramtelm([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Groldramvor(t *testing.T) {
	if got := Groldramvor([]float64{3, -7, 2, 4}, 3); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Groldramvor([]float64{1, 2, 3}, 3); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Groldramvor([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yethstel(t *testing.T) {
	if got := Yethstel([]float64{3, -7, 2, 4}, 4); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Yethstel([]float64{1, 2, 3}, 4); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Yethstel([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Plixhurn(t *testing.T) {
	if got := Plixhurn([]float64{3, -7, 2, 4}, 5); math.Abs(got-1.4) > 1e-9 { t.Fatalf("got %v want 1.4", got) }
	if got := Plixhurn([]float64{1, 2, 3}, 5); math.Abs(got-4.199999999999999) > 1e-9 { t.Fatalf("got %v want 4.199999999999999", got) }
	if got := Plixhurn([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Dramvorketh(t *testing.T) {
	if got := Dramvorketh([]float64{3, -7, 2, 4}, 2); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Dramvorketh([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Dramvorketh([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Sorvyethmex(t *testing.T) {
	if got := Sorvyethmex([]float64{3, -7, 2, 4}, 3); math.Abs(got-2.7) > 1e-9 { t.Fatalf("got %v want 2.7", got) }
	if got := Sorvyethmex([]float64{1, 2, 3}, 3); math.Abs(got-2.7) > 1e-9 { t.Fatalf("got %v want 2.7", got) }
	if got := Sorvyethmex([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Hurnxarn(t *testing.T) {
	if got := Hurnxarn([]float64{3, -7, 2, 4}, 4); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Hurnxarn([]float64{1, 2, 3}, 4); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Hurnxarn([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Telmbel(t *testing.T) {
	if got := Telmbel([]float64{3, -7, 2, 4}, 5); math.Abs(got-3.3000000000000003) > 1e-9 { t.Fatalf("got %v want 3.3000000000000003", got) }
	if got := Telmbel([]float64{1, 2, 3}, 5); math.Abs(got-2.2) > 1e-9 { t.Fatalf("got %v want 2.2", got) }
	if got := Telmbel([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nurbtrenith(t *testing.T) {
	if got := Nurbtrenith([]float64{3, -7, 2, 4}, 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Nurbtrenith([]float64{1, 2, 3}, 2); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Nurbtrenith([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Moxtelm(t *testing.T) {
	if got := Moxtelm([]float64{3, -7, 2, 4}, 3); math.Abs(got-1.7999999999999998) > 1e-9 { t.Fatalf("got %v want 1.7999999999999998", got) }
	if got := Moxtelm([]float64{1, 2, 3}, 3); math.Abs(got-1.2) > 1e-9 { t.Fatalf("got %v want 1.2", got) }
	if got := Moxtelm([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Paxdram(t *testing.T) {
	if got := Paxdram([]float64{3, -7, 2, 4}, 4); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Paxdram([]float64{1, 2, 3}, 4); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Paxdram([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Telmvix(t *testing.T) {
	if got := Telmvix([]float64{3, -7, 2, 4}, 5); math.Abs(got-1.6) > 1e-9 { t.Fatalf("got %v want 1.6", got) }
	if got := Telmvix([]float64{1, 2, 3}, 5); math.Abs(got-0.8) > 1e-9 { t.Fatalf("got %v want 0.8", got) }
	if got := Telmvix([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vixtrenulv(t *testing.T) {
	if got := Vixtrenulv([]float64{3, -7, 2, 4}, 2); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Vixtrenulv([]float64{1, 2, 3}, 2); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Vixtrenulv([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Skenithmox(t *testing.T) {
	if got := Skenithmox([]float64{3, -7, 2, 4}, 3); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Skenithmox([]float64{1, 2, 3}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Skenithmox([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Dramwex(t *testing.T) {
	if got := Dramwex([]float64{3, -7, 2, 4}, 4); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Dramwex([]float64{1, 2, 3}, 4); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Dramwex([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Quenpaxyul(t *testing.T) {
	if got := Quenpaxyul([]float64{3, -7, 2, 4}, 5); math.Abs(got-1.5) > 1e-9 { t.Fatalf("got %v want 1.5", got) }
	if got := Quenpaxyul([]float64{1, 2, 3}, 5); math.Abs(got-1.5) > 1e-9 { t.Fatalf("got %v want 1.5", got) }
	if got := Quenpaxyul([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Loryulnurb(t *testing.T) {
	if got := Loryulnurb([]float64{3, -7, 2, 4}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Loryulnurb([]float64{1, 2, 3}, 2); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Loryulnurb([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulmexwelm(t *testing.T) {
	if got := Yulmexwelm([]float64{3, -7, 2, 4}, 3); math.Abs(got-2.0999999999999996) > 1e-9 { t.Fatalf("got %v want 2.0999999999999996", got) }
	if got := Yulmexwelm([]float64{1, 2, 3}, 3); math.Abs(got-2.0999999999999996) > 1e-9 { t.Fatalf("got %v want 2.0999999999999996", got) }
	if got := Yulmexwelm([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Welmulvyeth(t *testing.T) {
	if got := Welmulvyeth([]float64{3, -7, 2, 4}, 4); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Welmulvyeth([]float64{1, 2, 3}, 4); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Welmulvyeth([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ulvplixyeth(t *testing.T) {
	if got := Ulvplixyeth([]float64{3, -7, 2, 4}, 5); math.Abs(got-2.7) > 1e-9 { t.Fatalf("got %v want 2.7", got) }
	if got := Ulvplixyeth([]float64{1, 2, 3}, 5); math.Abs(got-1.8) > 1e-9 { t.Fatalf("got %v want 1.8", got) }
	if got := Ulvplixyeth([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nurbwelmhurn(t *testing.T) {
	if got := Nurbwelmhurn([]float64{3, -7, 2, 4}, 2); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Nurbwelmhurn([]float64{1, 2, 3}, 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Nurbwelmhurn([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nethpax(t *testing.T) {
	if got := Nethpax([]float64{3, -7, 2, 4}, 3); math.Abs(got-3.3000000000000003) > 1e-9 { t.Fatalf("got %v want 3.3000000000000003", got) }
	if got := Nethpax([]float64{1, 2, 3}, 3); math.Abs(got-2.2) > 1e-9 { t.Fatalf("got %v want 2.2", got) }
	if got := Nethpax([]float64{}, 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Trenwelm(t *testing.T) {
	if got := Trenwelm([]float64{3, -7, 2, 4}, 4); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Trenwelm([]float64{1, 2, 3}, 4); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Trenwelm([]float64{}, 4); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Stelgrolvix(t *testing.T) {
	if got := Stelgrolvix([]float64{3, -7, 2, 4}, 5); math.Abs(got-1.2) > 1e-9 { t.Fatalf("got %v want 1.2", got) }
	if got := Stelgrolvix([]float64{1, 2, 3}, 5); math.Abs(got-0.6) > 1e-9 { t.Fatalf("got %v want 0.6", got) }
	if got := Stelgrolvix([]float64{}, 5); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ombrmex(t *testing.T) {
	if got := Ombrmex([]float64{3, -7, 2, 4}, 2); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Ombrmex([]float64{1, 2, 3}, 2); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Ombrmex([]float64{}, 2); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithtrensorv(t *testing.T) {
	if got := Ithtrensorv([]float64{3, -7, 2}, 3); math.Abs(got-(-8)) > 1e-9 { t.Fatalf("got %v want -8", got) }
	if got := Ithtrensorv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ithtrensorv([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Welmzarntren(t *testing.T) {
	if got := Welmzarntren([]float64{3, -7, 2}, 4); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Welmzarntren([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Welmzarntren([]float64{-2, 4, -1, 5}, 4); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Wexnurbpran(t *testing.T) {
	if got := Wexnurbpran([]float64{3, -7, 2}, 5); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Wexnurbpran([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Wexnurbpran([]float64{-2, 4, -1, 5}, 5); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Ulvwexxarn(t *testing.T) {
	if got := Ulvwexxarn([]float64{3, -7, 2}, 6); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Ulvwexxarn([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ulvwexxarn([]float64{-2, 4, -1, 5}, 6); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Skenyulwelm(t *testing.T) {
	if got := Skenyulwelm([]float64{3, -7, 2}, 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Skenyulwelm([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Skenyulwelm([]float64{-2, 4, -1, 5}, 2); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Xarnkivsken(t *testing.T) {
	if got := Xarnkivsken([]float64{3, -7, 2}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Xarnkivsken([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Xarnkivsken([]float64{-2, 4, -1, 5}, 3); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
}

func Test_Plixithquel(t *testing.T) {
	if got := Plixithquel([]float64{3, -7, 2}, 4); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Plixithquel([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Plixithquel([]float64{-2, 4, -1, 5}, 4); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Brindramhurn(t *testing.T) {
	if got := Brindramhurn([]float64{3, -7, 2}, 5); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Brindramhurn([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Brindramhurn([]float64{-2, 4, -1, 5}, 5); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Ithquen(t *testing.T) {
	if got := Ithquen([]float64{3, -7, 2}, 6); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Ithquen([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ithquen([]float64{-2, 4, -1, 5}, 6); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Trengrol(t *testing.T) {
	if got := Trengrol([]float64{3, -7, 2}, 2); math.Abs(got-(-5)) > 1e-9 { t.Fatalf("got %v want -5", got) }
	if got := Trengrol([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Trengrol([]float64{-2, 4, -1, 5}, 2); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
}

func Test_Ithquel(t *testing.T) {
	if got := Ithquel([]float64{3, -7, 2}, 3); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Ithquel([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Ithquel([]float64{-2, 4, -1, 5}, 3); math.Abs(got-10) > 1e-9 { t.Fatalf("got %v want 10", got) }
}

func Test_Quenwelmquen(t *testing.T) {
	if got := Quenwelmquen([]float64{3, -7, 2}, 4); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Quenwelmquen([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Quenwelmquen([]float64{-2, 4, -1, 5}, 4); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Zilmstel(t *testing.T) {
	if got := Zilmstel([]float64{3, -7, 2}, 5); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Zilmstel([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Zilmstel([]float64{-2, 4, -1, 5}, 5); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Stelquenhurn(t *testing.T) {
	if got := Stelquenhurn([]float64{3, -7, 2}, 6); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Stelquenhurn([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Stelquenhurn([]float64{-2, 4, -1, 5}, 6); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Plixsorv(t *testing.T) {
	if got := Plixsorv([]float64{3, -7, 2}, 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Plixsorv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Plixsorv([]float64{-2, 4, -1, 5}, 2); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Moxhurnorv(t *testing.T) {
	if got := Moxhurnorv([]float64{3, -7, 2}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Moxhurnorv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Moxhurnorv([]float64{-2, 4, -1, 5}, 3); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
}

func Test_Wexzilmmex(t *testing.T) {
	if got := Wexzilmmex([]float64{3, -7, 2}, 4); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Wexzilmmex([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Wexzilmmex([]float64{-2, 4, -1, 5}, 4); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Zarnorvstel(t *testing.T) {
	if got := Zarnorvstel([]float64{3, -7, 2}, 5); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Zarnorvstel([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Zarnorvstel([]float64{-2, 4, -1, 5}, 5); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
}

func Test_Nethulv(t *testing.T) {
	if got := Nethulv([]float64{3, -7, 2}, 6); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Nethulv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Nethulv([]float64{-2, 4, -1, 5}, 6); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
}

func Test_Vixyululv(t *testing.T) {
	if got := Vixyululv([]float64{3, -7, 2}, 2); math.Abs(got-(-5)) > 1e-9 { t.Fatalf("got %v want -5", got) }
	if got := Vixyululv([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Vixyululv([]float64{-2, 4, -1, 5}, 2); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
}

func Test_Trenhurngrol(t *testing.T) {
	if got := Trenhurngrol("ab9", 3); math.Abs(got-176.39999999999998) > 1e-9 { t.Fatalf("got %v want 176.39999999999998", got) }
	if got := Trenhurngrol("x1,y2", 3); math.Abs(got-268.79999999999995) > 1e-9 { t.Fatalf("got %v want 268.79999999999995", got) }
	if got := Trenhurngrol("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulquensken(t *testing.T) {
	if got := Yulquensken("ab9", 4); math.Abs(got-248) > 1e-9 { t.Fatalf("got %v want 248", got) }
	if got := Yulquensken("x1,y2", 4); math.Abs(got-380) > 1e-9 { t.Fatalf("got %v want 380", got) }
	if got := Yulquensken("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Moxombr(t *testing.T) {
	if got := Moxombr("ab9", 5); math.Abs(got-226.8) > 1e-9 { t.Fatalf("got %v want 226.8", got) }
	if got := Moxombr("x1,y2", 5); math.Abs(got-345.6) > 1e-9 { t.Fatalf("got %v want 345.6", got) }
	if got := Moxombr("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Belyulflep(t *testing.T) {
	if got := Belyulflep("ab9", 6); math.Abs(got-250) > 1e-9 { t.Fatalf("got %v want 250", got) }
	if got := Belyulflep("x1,y2", 6); math.Abs(got-382) > 1e-9 { t.Fatalf("got %v want 382", got) }
	if got := Belyulflep("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithxarnzarn(t *testing.T) {
	if got := Ithxarnzarn("ab9", 2); math.Abs(got-277.20000000000005) > 1e-9 { t.Fatalf("got %v want 277.20000000000005", got) }
	if got := Ithxarnzarn("x1,y2", 2); math.Abs(got-422.40000000000003) > 1e-9 { t.Fatalf("got %v want 422.40000000000003", got) }
	if got := Ithxarnzarn("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Lorbelwex(t *testing.T) {
	if got := Lorbelwex("ab9", 3); math.Abs(got-252) > 1e-9 { t.Fatalf("got %v want 252", got) }
	if got := Lorbelwex("x1,y2", 3); math.Abs(got-384) > 1e-9 { t.Fatalf("got %v want 384", got) }
	if got := Lorbelwex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Plixmox(t *testing.T) {
	if got := Plixmox("ab9", 4); math.Abs(got-151.2) > 1e-9 { t.Fatalf("got %v want 151.2", got) }
	if got := Plixmox("x1,y2", 4); math.Abs(got-230.39999999999998) > 1e-9 { t.Fatalf("got %v want 230.39999999999998", got) }
	if got := Plixmox("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zarnzarn(t *testing.T) {
	if got := Zarnzarn("ab9", 5); math.Abs(got-254) > 1e-9 { t.Fatalf("got %v want 254", got) }
	if got := Zarnzarn("x1,y2", 5); math.Abs(got-386) > 1e-9 { t.Fatalf("got %v want 386", got) }
	if got := Zarnzarn("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Quenmex(t *testing.T) {
	if got := Quenmex("ab9", 6); math.Abs(got-201.60000000000002) > 1e-9 { t.Fatalf("got %v want 201.60000000000002", got) }
	if got := Quenmex("x1,y2", 6); math.Abs(got-307.20000000000005) > 1e-9 { t.Fatalf("got %v want 307.20000000000005", got) }
	if got := Quenmex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithtelm(t *testing.T) {
	if got := Ithtelm("ab9", 2); math.Abs(got-256) > 1e-9 { t.Fatalf("got %v want 256", got) }
	if got := Ithtelm("x1,y2", 2); math.Abs(got-388) > 1e-9 { t.Fatalf("got %v want 388", got) }
	if got := Ithtelm("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grolgrolmex(t *testing.T) {
	if got := Grolgrolmex("ab9", 3); math.Abs(got-252) > 1e-9 { t.Fatalf("got %v want 252", got) }
	if got := Grolgrolmex("x1,y2", 3); math.Abs(got-384) > 1e-9 { t.Fatalf("got %v want 384", got) }
	if got := Grolgrolmex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zilmpran(t *testing.T) {
	if got := Zilmpran("ab9", 4); math.Abs(got-247) > 1e-9 { t.Fatalf("got %v want 247", got) }
	if got := Zilmpran("x1,y2", 4); math.Abs(got-379) > 1e-9 { t.Fatalf("got %v want 379", got) }
	if got := Zilmpran("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Quenpax(t *testing.T) {
	if got := Quenpax("ab9", 5); math.Abs(got-126) > 1e-9 { t.Fatalf("got %v want 126", got) }
	if got := Quenpax("x1,y2", 5); math.Abs(got-192) > 1e-9 { t.Fatalf("got %v want 192", got) }
	if got := Quenpax("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Wexmoxulv(t *testing.T) {
	if got := Wexmoxulv("ab9", 6); math.Abs(got-249) > 1e-9 { t.Fatalf("got %v want 249", got) }
	if got := Wexmoxulv("x1,y2", 6); math.Abs(got-381) > 1e-9 { t.Fatalf("got %v want 381", got) }
	if got := Wexmoxulv("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Lorneth(t *testing.T) {
	if got := Lorneth("ab9", 2); math.Abs(got-176.39999999999998) > 1e-9 { t.Fatalf("got %v want 176.39999999999998", got) }
	if got := Lorneth("x1,y2", 2); math.Abs(got-268.79999999999995) > 1e-9 { t.Fatalf("got %v want 268.79999999999995", got) }
	if got := Lorneth("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yethpranyeth(t *testing.T) {
	if got := Yethpranyeth("ab9", 3); math.Abs(got-251) > 1e-9 { t.Fatalf("got %v want 251", got) }
	if got := Yethpranyeth("x1,y2", 3); math.Abs(got-383) > 1e-9 { t.Fatalf("got %v want 383", got) }
	if got := Yethpranyeth("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nethkiv(t *testing.T) {
	if got := Nethkiv("ab9", 4); math.Abs(got-226.8) > 1e-9 { t.Fatalf("got %v want 226.8", got) }
	if got := Nethkiv("x1,y2", 4); math.Abs(got-345.6) > 1e-9 { t.Fatalf("got %v want 345.6", got) }
	if got := Nethkiv("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Daxquen(t *testing.T) {
	if got := Daxquen("ab9", 5); math.Abs(got-253) > 1e-9 { t.Fatalf("got %v want 253", got) }
	if got := Daxquen("x1,y2", 5); math.Abs(got-385) > 1e-9 { t.Fatalf("got %v want 385", got) }
	if got := Daxquen("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Paxsorvombr(t *testing.T) {
	if got := Paxsorvombr("ab9", 6); math.Abs(got-277.20000000000005) > 1e-9 { t.Fatalf("got %v want 277.20000000000005", got) }
	if got := Paxsorvombr("x1,y2", 6); math.Abs(got-422.40000000000003) > 1e-9 { t.Fatalf("got %v want 422.40000000000003", got) }
	if got := Paxsorvombr("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Plixkethorv(t *testing.T) {
	if got := Plixkethorv("ab9", 2); math.Abs(got-255) > 1e-9 { t.Fatalf("got %v want 255", got) }
	if got := Plixkethorv("x1,y2", 2); math.Abs(got-387) > 1e-9 { t.Fatalf("got %v want 387", got) }
	if got := Plixkethorv("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Quenpaxkiv(t *testing.T) {
	if got := Quenpaxkiv("ab9", 3); math.Abs(got-5.3999999999999995) > 1e-9 { t.Fatalf("got %v want 5.3999999999999995", got) }
	if got := Quenpaxkiv("x1,y2", 3); math.Abs(got-1.7999999999999998) > 1e-9 { t.Fatalf("got %v want 1.7999999999999998", got) }
	if got := Quenpaxkiv("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Mexxarn(t *testing.T) {
	if got := Mexxarn("ab9", 4); math.Abs(got-14) > 1e-9 { t.Fatalf("got %v want 14", got) }
	if got := Mexxarn("x1,y2", 4); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Mexxarn("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Lorvex(t *testing.T) {
	if got := Lorvex("ab9", 5); math.Abs(got-7.2) > 1e-9 { t.Fatalf("got %v want 7.2", got) }
	if got := Lorvex("x1,y2", 5); math.Abs(got-2.4000000000000004) > 1e-9 { t.Fatalf("got %v want 2.4000000000000004", got) }
	if got := Lorvex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Wexdramtelm(t *testing.T) {
	if got := Wexdramtelm("ab9", 6); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Wexdramtelm("x1,y2", 6); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Wexdramtelm("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Telmgrol(t *testing.T) {
	if got := Telmgrol("ab9", 2); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Telmgrol("x1,y2", 2); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Telmgrol("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Brinkivdram(t *testing.T) {
	if got := Brinkivdram("ab9", 3); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Brinkivdram("x1,y2", 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Brinkivdram("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ulvmex(t *testing.T) {
	if got := Ulvmex("ab9", 4); math.Abs(got-4.5) > 1e-9 { t.Fatalf("got %v want 4.5", got) }
	if got := Ulvmex("x1,y2", 4); math.Abs(got-1.5) > 1e-9 { t.Fatalf("got %v want 1.5", got) }
	if got := Ulvmex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Lordax(t *testing.T) {
	if got := Lordax("ab9", 5); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Lordax("x1,y2", 5); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Lordax("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithmex(t *testing.T) {
	if got := Ithmex("ab9", 6); math.Abs(got-6.3) > 1e-9 { t.Fatalf("got %v want 6.3", got) }
	if got := Ithmex("x1,y2", 6); math.Abs(got-2.0999999999999996) > 1e-9 { t.Fatalf("got %v want 2.0999999999999996", got) }
	if got := Ithmex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Mexmex(t *testing.T) {
	if got := Mexmex("ab9", 2); math.Abs(got-11) > 1e-9 { t.Fatalf("got %v want 11", got) }
	if got := Mexmex("x1,y2", 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Mexmex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Belorv(t *testing.T) {
	if got := Belorv("ab9", 3); math.Abs(got-8.1) > 1e-9 { t.Fatalf("got %v want 8.1", got) }
	if got := Belorv("x1,y2", 3); math.Abs(got-2.7) > 1e-9 { t.Fatalf("got %v want 2.7", got) }
	if got := Belorv("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Ithvixwex(t *testing.T) {
	if got := Ithvixwex("ab9", 4); math.Abs(got-13) > 1e-9 { t.Fatalf("got %v want 13", got) }
	if got := Ithvixwex("x1,y2", 4); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Ithvixwex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Plixplixvex(t *testing.T) {
	if got := Plixplixvex("ab9", 5); math.Abs(got-9.9) > 1e-9 { t.Fatalf("got %v want 9.9", got) }
	if got := Plixplixvex("x1,y2", 5); math.Abs(got-3.3000000000000003) > 1e-9 { t.Fatalf("got %v want 3.3000000000000003", got) }
	if got := Plixplixvex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Wexzilm(t *testing.T) {
	if got := Wexzilm("ab9", 6); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Wexzilm("x1,y2", 6); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Wexzilm("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grolbrin(t *testing.T) {
	if got := Grolbrin("ab9", 2); math.Abs(got-5.3999999999999995) > 1e-9 { t.Fatalf("got %v want 5.3999999999999995", got) }
	if got := Grolbrin("x1,y2", 2); math.Abs(got-1.7999999999999998) > 1e-9 { t.Fatalf("got %v want 1.7999999999999998", got) }
	if got := Grolbrin("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Drammox(t *testing.T) {
	if got := Drammox("ab9", 3); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Drammox("x1,y2", 3); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Drammox("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Belgrol(t *testing.T) {
	if got := Belgrol("ab9", 4); math.Abs(got-7.2) > 1e-9 { t.Fatalf("got %v want 7.2", got) }
	if got := Belgrol("x1,y2", 4); math.Abs(got-2.4000000000000004) > 1e-9 { t.Fatalf("got %v want 2.4000000000000004", got) }
	if got := Belgrol("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Orvkethtelm(t *testing.T) {
	if got := Orvkethtelm("ab9", 5); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Orvkethtelm("x1,y2", 5); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Orvkethtelm("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulsken(t *testing.T) {
	if got := Yulsken("ab9", 6); math.Abs(got-9) > 1e-9 { t.Fatalf("got %v want 9", got) }
	if got := Yulsken("x1,y2", 6); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Yulsken("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zilmulvwex(t *testing.T) {
	if got := Zilmulvwex("ab9", 2); math.Abs(got-10) > 1e-9 { t.Fatalf("got %v want 10", got) }
	if got := Zilmulvwex("x1,y2", 2); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Zilmulvwex("", 1); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Belkiv(t *testing.T) {
	if got := Belkiv([]float64{3, -7, 2}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Belkiv([]float64{1}); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Belkiv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Kethvor(t *testing.T) {
	if got := Kethvor([]float64{3, -7, 2}); math.Abs(got-2.3333333333333335) > 1e-9 { t.Fatalf("got %v want 2.3333333333333335", got) }
	if got := Kethvor([]float64{1}); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Kethvor([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vexkiv(t *testing.T) {
	if got := Vexkiv([]float64{3, -7, 2}); math.Abs(got-16) > 1e-9 { t.Fatalf("got %v want 16", got) }
	if got := Vexkiv([]float64{1}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Vexkiv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Vexmexflep(t *testing.T) {
	if got := Vexmexflep([]float64{3, -7, 2}); math.Abs(got-3.2) > 1e-9 { t.Fatalf("got %v want 3.2", got) }
	if got := Vexmexflep([]float64{1}); math.Abs(got-0.8) > 1e-9 { t.Fatalf("got %v want 0.8", got) }
	if got := Vexmexflep([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Wexbrin(t *testing.T) {
	if got := Wexbrin([]float64{3, -7, 2}); math.Abs(got-7) > 1e-9 { t.Fatalf("got %v want 7", got) }
	if got := Wexbrin([]float64{1}); math.Abs(got-(-4)) > 1e-9 { t.Fatalf("got %v want -4", got) }
	if got := Wexbrin([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Hurnvorombr(t *testing.T) {
	if got := Hurnvorombr([]float64{3, -7, 2}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Hurnvorombr([]float64{1}); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Hurnvorombr([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Dramstelkiv(t *testing.T) {
	if got := Dramstelkiv([]float64{3, -7, 2}); math.Abs(got-13.200000000000001) > 1e-9 { t.Fatalf("got %v want 13.200000000000001", got) }
	if got := Dramstelkiv([]float64{1}); math.Abs(got-1.1) > 1e-9 { t.Fatalf("got %v want 1.1", got) }
	if got := Dramstelkiv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Skenombrdax(t *testing.T) {
	if got := Skenombrdax([]float64{3, -7, 2}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Skenombrdax([]float64{1}); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Skenombrdax([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yulpranorv(t *testing.T) {
	if got := Yulpranorv([]float64{3, -7, 2}); math.Abs(got-11) > 1e-9 { t.Fatalf("got %v want 11", got) }
	if got := Yulpranorv([]float64{1}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Yulpranorv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Grolhurn(t *testing.T) {
	if got := Grolhurn([]float64{3, -7, 2}); math.Abs(got-2.8) > 1e-9 { t.Fatalf("got %v want 2.8", got) }
	if got := Grolhurn([]float64{1}); math.Abs(got-0.7) > 1e-9 { t.Fatalf("got %v want 0.7", got) }
	if got := Grolhurn([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Daxrilm(t *testing.T) {
	if got := Daxrilm([]float64{3, -7, 2}); math.Abs(got-13) > 1e-9 { t.Fatalf("got %v want 13", got) }
	if got := Daxrilm([]float64{1}); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Daxrilm([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Nurbwex(t *testing.T) {
	if got := Nurbwex([]float64{3, -7, 2}); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Nurbwex([]float64{1}); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Nurbwex([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Daxdaxkiv(t *testing.T) {
	if got := Daxdaxkiv([]float64{3, -7, 2}); math.Abs(got-12) > 1e-9 { t.Fatalf("got %v want 12", got) }
	if got := Daxdaxkiv([]float64{1}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Daxdaxkiv([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Stelxarn(t *testing.T) {
	if got := Stelxarn([]float64{3, -7, 2}); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Stelxarn([]float64{1}); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Stelxarn([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Quelsorvbel(t *testing.T) {
	if got := Quelsorvbel([]float64{3, -7, 2}); math.Abs(got-17) > 1e-9 { t.Fatalf("got %v want 17", got) }
	if got := Quelsorvbel([]float64{1}); math.Abs(got-6) > 1e-9 { t.Fatalf("got %v want 6", got) }
	if got := Quelsorvbel([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Lorquel(t *testing.T) {
	if got := Lorquel([]float64{3, -7, 2}); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
	if got := Lorquel([]float64{1}); math.Abs(got-0.6) > 1e-9 { t.Fatalf("got %v want 0.6", got) }
	if got := Lorquel([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Trenxarnwelm(t *testing.T) {
	if got := Trenxarnwelm([]float64{3, -7, 2}); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Trenxarnwelm([]float64{1}); math.Abs(got-(-3)) > 1e-9 { t.Fatalf("got %v want -3", got) }
	if got := Trenxarnwelm([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Drambrin(t *testing.T) {
	if got := Drambrin([]float64{3, -7, 2}); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Drambrin([]float64{1}); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Drambrin([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Yethkethneth(t *testing.T) {
	if got := Yethkethneth([]float64{3, -7, 2}); math.Abs(got-10.8) > 1e-9 { t.Fatalf("got %v want 10.8", got) }
	if got := Yethkethneth([]float64{1}); math.Abs(got-0.9) > 1e-9 { t.Fatalf("got %v want 0.9", got) }
	if got := Yethkethneth([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Dramtrenyul(t *testing.T) {
	if got := Dramtrenyul([]float64{3, -7, 2}); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Dramtrenyul([]float64{1}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
	if got := Dramtrenyul([]float64{}); math.Abs(got-0) > 1e-9 { t.Fatalf("got %v want 0", got) }
}

func Test_Zarnzilmlor(t *testing.T) {
	if got := Zarnzilmlor([]float64{3, -7, 2}, 3); math.Abs(got-8) > 1e-9 { t.Fatalf("got %v want 8", got) }
	if got := Zarnzilmlor([]float64{1}, 3); math.Abs(got-1) > 1e-9 { t.Fatalf("got %v want 1", got) }
	if got := Zarnzilmlor([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
}

func Test_Zilmrilmxarn(t *testing.T) {
	if got := Zilmrilmxarn([]float64{3, -7, 2}, 4); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
	if got := Zilmrilmxarn([]float64{1}, 3); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Zilmrilmxarn([]float64{-2, 4, -1, 5}, 4); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
}

func Test_Trenorvketh(t *testing.T) {
	if got := Trenorvketh([]float64{3, -7, 2}, 5); math.Abs(got-(-0.4)) > 1e-9 { t.Fatalf("got %v want -0.4", got) }
	if got := Trenorvketh([]float64{1}, 3); math.Abs(got-0.2) > 1e-9 { t.Fatalf("got %v want 0.2", got) }
	if got := Trenorvketh([]float64{-2, 4, -1, 5}, 5); math.Abs(got-(-1.2)) > 1e-9 { t.Fatalf("got %v want -1.2", got) }
}

func Test_Queldram(t *testing.T) {
	if got := Queldram([]float64{3, -7, 2}, 6); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Queldram([]float64{1}, 3); math.Abs(got-0.16666666666666666) > 1e-9 { t.Fatalf("got %v want 0.16666666666666666", got) }
	if got := Queldram([]float64{-2, 4, -1, 5}, 6); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
}

func Test_Quenmexvex(t *testing.T) {
	if got := Quenmexvex([]float64{3, -7, 2}, 2); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Quenmexvex([]float64{1}, 3); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Quenmexvex([]float64{-2, 4, -1, 5}, 2); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Moxulvneth(t *testing.T) {
	if got := Moxulvneth([]float64{3, -7, 2}, 3); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Moxulvneth([]float64{1}, 3); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Moxulvneth([]float64{-2, 4, -1, 5}, 3); math.Abs(got-2.3333333333333335) > 1e-9 { t.Fatalf("got %v want 2.3333333333333335", got) }
}

func Test_Daxtrenulv(t *testing.T) {
	if got := Daxtrenulv([]float64{3, -7, 2}, 4); math.Abs(got-(-0.5)) > 1e-9 { t.Fatalf("got %v want -0.5", got) }
	if got := Daxtrenulv([]float64{1}, 3); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
	if got := Daxtrenulv([]float64{-2, 4, -1, 5}, 4); math.Abs(got-(-1.5)) > 1e-9 { t.Fatalf("got %v want -1.5", got) }
}

func Test_Zilmwelmtelm(t *testing.T) {
	if got := Zilmwelmtelm([]float64{3, -7, 2}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
	if got := Zilmwelmtelm([]float64{1}, 3); math.Abs(got-0.2) > 1e-9 { t.Fatalf("got %v want 0.2", got) }
	if got := Zilmwelmtelm([]float64{-2, 4, -1, 5}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
}

func Test_Quenrilm(t *testing.T) {
	if got := Quenrilm([]float64{3, -7, 2}, 6); math.Abs(got-(-0.3333333333333333)) > 1e-9 { t.Fatalf("got %v want -0.3333333333333333", got) }
	if got := Quenrilm([]float64{1}, 3); math.Abs(got-0.16666666666666666) > 1e-9 { t.Fatalf("got %v want 0.16666666666666666", got) }
	if got := Quenrilm([]float64{-2, 4, -1, 5}, 6); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Sorvgrol(t *testing.T) {
	if got := Sorvgrol([]float64{3, -7, 2}, 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Sorvgrol([]float64{1}, 3); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Sorvgrol([]float64{-2, 4, -1, 5}, 2); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
}

func Test_Telmwexulv(t *testing.T) {
	if got := Telmwexulv([]float64{3, -7, 2}, 3); math.Abs(got-(-0.6666666666666666)) > 1e-9 { t.Fatalf("got %v want -0.6666666666666666", got) }
	if got := Telmwexulv([]float64{1}, 3); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Telmwexulv([]float64{-2, 4, -1, 5}, 3); math.Abs(got-(-0.3333333333333333)) > 1e-9 { t.Fatalf("got %v want -0.3333333333333333", got) }
}

func Test_Orvzilm(t *testing.T) {
	if got := Orvzilm([]float64{3, -7, 2}, 4); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
	if got := Orvzilm([]float64{1}, 3); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
	if got := Orvzilm([]float64{-2, 4, -1, 5}, 4); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
}

func Test_Lorvix(t *testing.T) {
	if got := Lorvix([]float64{3, -7, 2}, 5); math.Abs(got-(-0.4)) > 1e-9 { t.Fatalf("got %v want -0.4", got) }
	if got := Lorvix([]float64{1}, 3); math.Abs(got-0.2) > 1e-9 { t.Fatalf("got %v want 0.2", got) }
	if got := Lorvix([]float64{-2, 4, -1, 5}, 5); math.Abs(got-(-1.2)) > 1e-9 { t.Fatalf("got %v want -1.2", got) }
}

func Test_Vixulv(t *testing.T) {
	if got := Vixulv([]float64{3, -7, 2}, 6); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
	if got := Vixulv([]float64{1}, 3); math.Abs(got-0.16666666666666666) > 1e-9 { t.Fatalf("got %v want 0.16666666666666666", got) }
	if got := Vixulv([]float64{-2, 4, -1, 5}, 6); math.Abs(got-2) > 1e-9 { t.Fatalf("got %v want 2", got) }
}

func Test_Stelplixstel(t *testing.T) {
	if got := Stelplixstel([]float64{3, -7, 2}, 2); math.Abs(got-(-2)) > 1e-9 { t.Fatalf("got %v want -2", got) }
	if got := Stelplixstel([]float64{1}, 3); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Stelplixstel([]float64{-2, 4, -1, 5}, 2); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Kivyethpax(t *testing.T) {
	if got := Kivyethpax([]float64{3, -7, 2}, 3); math.Abs(got-4) > 1e-9 { t.Fatalf("got %v want 4", got) }
	if got := Kivyethpax([]float64{1}, 3); math.Abs(got-0.3333333333333333) > 1e-9 { t.Fatalf("got %v want 0.3333333333333333", got) }
	if got := Kivyethpax([]float64{-2, 4, -1, 5}, 3); math.Abs(got-2.3333333333333335) > 1e-9 { t.Fatalf("got %v want 2.3333333333333335", got) }
}

func Test_Flepkiv(t *testing.T) {
	if got := Flepkiv([]float64{3, -7, 2}, 4); math.Abs(got-(-0.5)) > 1e-9 { t.Fatalf("got %v want -0.5", got) }
	if got := Flepkiv([]float64{1}, 3); math.Abs(got-0.25) > 1e-9 { t.Fatalf("got %v want 0.25", got) }
	if got := Flepkiv([]float64{-2, 4, -1, 5}, 4); math.Abs(got-(-1.5)) > 1e-9 { t.Fatalf("got %v want -1.5", got) }
}

func Test_Orvbelyul(t *testing.T) {
	if got := Orvbelyul([]float64{3, -7, 2}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
	if got := Orvbelyul([]float64{1}, 3); math.Abs(got-0.2) > 1e-9 { t.Fatalf("got %v want 0.2", got) }
	if got := Orvbelyul([]float64{-2, 4, -1, 5}, 5); math.Abs(got-2.4) > 1e-9 { t.Fatalf("got %v want 2.4", got) }
}

func Test_Daxwelm(t *testing.T) {
	if got := Daxwelm([]float64{3, -7, 2}, 6); math.Abs(got-(-0.3333333333333333)) > 1e-9 { t.Fatalf("got %v want -0.3333333333333333", got) }
	if got := Daxwelm([]float64{1}, 3); math.Abs(got-0.16666666666666666) > 1e-9 { t.Fatalf("got %v want 0.16666666666666666", got) }
	if got := Daxwelm([]float64{-2, 4, -1, 5}, 6); math.Abs(got-(-1)) > 1e-9 { t.Fatalf("got %v want -1", got) }
}

func Test_Nurbulv(t *testing.T) {
	if got := Nurbulv([]float64{3, -7, 2}, 2); math.Abs(got-5) > 1e-9 { t.Fatalf("got %v want 5", got) }
	if got := Nurbulv([]float64{1}, 3); math.Abs(got-0.5) > 1e-9 { t.Fatalf("got %v want 0.5", got) }
	if got := Nurbulv([]float64{-2, 4, -1, 5}, 2); math.Abs(got-3) > 1e-9 { t.Fatalf("got %v want 3", got) }
}

