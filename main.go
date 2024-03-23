package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

// type layer func([]float32, [][]float32, []float32, string) []float32
//
//	type NeuralNetwork struct {
//		layer []layer
//		w     [][][]float32
//		b     []float32
//		lr    float32
//	}
func main() {

	t := Tensor{}
	t.readNpy("data/test_at9.npy")
	fmt.Println(t.at(3, 4, 0, 1, 2))
	fmt.Println(t.shape)
	// t.at(0, 0)
	// t.at(0, 1)
	// t.at(1, 0)
	// t.at(1, 1)
	// t.at(2, 0)
	// t.at(2, 1)
	// fmt.Println(in)
	// fmt.Printf("in:\n%v\n", in)
	// fmt.Printf("w:\n%v\n", w)
	// dense(in, w0.(*Dense), b, "")
}

// func dense(in *Dense, w *Dense, b *Dense, act string) *Dense {
//
// 	// var activ activation
// 	// switch act {
// 	// case "relu":
// 	// 	activ = relu
// 	// case "sigmoid":
// 	// 	activ = sigmoid
// 	// default:
// 	// 	activ = pass
// 	// }
// 	fmt.Println(in)
// 	fmt.Printf("in:\n%v\n", in)
// 	fmt.Printf("w:\n%v\n", w)
// 	// w0, _ := w.Slice(S(0), S(0), nil)
// 	// fmt.Printf("w0:\n%v\n", w0)
// 	// w1, _ := w.Slice(S(0), nil, S(0))
// 	// fmt.Printf("w1:\n%v\n", w1)
// 	fmt.Printf("w:\n%v\n", w)
// 	// out := make([]float32, len(w))
// 	fmt.Printf("b:\n%v\n", b)
// 	fmt.Println(w.Shape())
// 	for i := 0; i < w.Shape()[0]; i++ {
// 		tempw, _ := w.Slice(S(i), nil)
// 		fmt.Printf("tempw:\n%v\n", tempw)
// 		tempm, _ := in.Mul(tempw.(*Dense))
// 		fmt.Printf("tempw:\n%v\n", tempm)
// 		temps, _ := tempm.Sum()
// 		fmt.Printf("temps:\n%v\n", temps)
// 		temp := float32(0)
// 		tempsi, _ := temps.At(0)
// 		temp += tempsi
// 	}
// 	// out := in
// 	return w
// }

func denseBP(w [][]float32, b []float32, err []float32, input []float32, output []float32, lr float32) []float32 {

	errout := make([]float32, len(input))
	for i := range w {
		delta := err[i] * sigmoidDer(output[i])
		for j := range w[i] {
			errout[j] += delta * w[i][j]
			w[i][j] = w[i][j] - lr*delta*input[j]

		}
		b[i] = b[i] - lr*delta*1
	}
	return errout
}

// func train() {
//
// 	lr := float32(0.5)
// 	input := []float32{0.05, 0.1}
// 	w1 := [][]float32{{0.15, 0.25}, {0.2, 0.3}}
// 	w2 := [][]float32{{0.40, 0.5}, {0.45, 0.55}}
// 	bias1 := []float32{0.35, 0.35}
// 	bias2 := []float32{0.6, 0.6}
// 	target := []float32{0.01, 0.99}
//
// 	out1 := dense(input, w1, bias1, "sigmoid")
// 	out2 := dense(out1, w2, bias2, "sigmoid")
//
// 	// dout2 := make([]float32, len(out2))
//
// 	fmt.Println("Layer 2:")
// 	err2 := loss(out2, target)
// 	dout2 := denseBP(w2, bias2, err2, out1, out2, lr)
// 	fmt.Println(dout2)
// 	fmt.Println(w2)
// 	fmt.Println(bias2)
// 	fmt.Println("Layer 1:")
// 	_ = denseBP(w1, bias1, dout2, input, out1, lr)
// 	fmt.Println(w1)
// 	fmt.Println(bias1)
// }

func loss(arr []float32, target []float32) []float32 {
	resArr := make([]float32, len(arr))
	for i := range arr {
		temp := target[i] - arr[i]
		resArr[i] = -temp
	}
	return resArr
}

func sumArr(arr []float32) float32 {
	result := float32(0.0)
	for i := range arr {
		result += arr[i]
	}
	return result
}

// func conv2d(in, wgt tensor, relu, padding bool) tensor {
//
// 	return nil
// }

func max(x, y float32) float32 {
	if x > y {
		return x
	}
	return y
}

type activation func(float32) float32

func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(x)*(-1))))
}
func relu(x float32) float32 {
	return max(0, x)
}
func pass(x float32) float32 {
	return x
}

func sigmoidDer(x float32) float32 {
	return x * (1 - x)
}

func activate(x, b float32, fn activation) float32 {
	raw := x + b
	return fn(raw)
}
func readFile(fileName string) []string {

	inRead, err := os.ReadFile(fileName)
	if err != nil {
		panic(err)
	}

	str := string(inRead)
	str = strings.ReplaceAll(str, "\n", " ")
	strArr := make([]string, 0)
	strArr = strings.Split(str, " ")

	return strArr
}

func strArrToFlat(s []string, x, y int) [][]float32 {
	if len(s) < x*y {
		log.Fatalf("func strArrToTensor: Input string length %d smaller than array size %d.\n", len(s), x*y)
	}
	si := 0
	arr := make([][]float32, x)
	for i := 0; i < x; i++ {
		arr[i] = make([]float32, y)
		for j := 0; j < y; j++ {
			if len(s[si]) > 0 {
				temp, err := strconv.ParseFloat(s[si], 32)
				if err != nil {
					log.Fatalln(err, i, j, temp, s[si], len(s[si]))
				}
				arr[i][j] = float32(temp)
			}
			si++
		}
	}
	return arr
}
