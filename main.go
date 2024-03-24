package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

func simpleNet() {

	w1 := Tensor{}
	w1.readNpy("test_data/w1.npy")
	w2 := Tensor{}
	w2.readNpy("test_data/w2.npy")
	b1 := Tensor{}
	b1.readNpy("test_data/b1.npy")
	b2 := Tensor{}
	b2.readNpy("test_data/b2.npy")
	in := Tensor{}
	in.readNpy("test_data/in.npy")
	gout := Tensor{}
	gout.readNpy("test_data/out.npy")
	target := Tensor{
		shape: []int{2},
		arr:   []float32{0.01, 0.99},
	}

	dout1 := dense(in, w1, b1, "sigmoid")
	dout2 := dense(dout1, w2, b2, "sigmoid")
	fmt.Println(dout2)

	outLoss := loss(dout2, target)
	fmt.Println(outLoss)
	totLoss := totalLoss(dout2, target)
	fmt.Println(totLoss)
	l2Loss := denseBP(w2, b2, outLoss, dout1, dout2, float32(0.5))
	fmt.Println(w2)
	fmt.Println(b2)
	fmt.Println(l2Loss)
	l1Loss := denseBP(w1, b1, l2Loss, in, dout1, float32(0.5))
	fmt.Println(w1)
	fmt.Println(b1)
	fmt.Println(l1Loss)

}

func main() {

	simpleNet()
}

func dense(in Tensor, w Tensor, b Tensor, act string) Tensor {

	var activ activation
	switch act {
	case "relu":
		activ = relu
	case "sigmoid":
		activ = sigmoid
	default:
		activ = pass
	}

	out := Tensor{}
	out.zero(w.shape[1])

	for i := 0; i < w.shape[0]; i++ {
		temp := float32(0.0)
		for j := 0; j < w.shape[1]; j++ {
			temp += w.at(j, i) * in.at(j)
		}
		out.set(activ(temp+b.at(i)), i)
	}
	return out
}

func denseBP(w Tensor, b Tensor, err Tensor, input Tensor, output Tensor, lr float32) Tensor {

	errout := Tensor{}
	errout.zero(output.shape[0])
	for i := 0; i < w.shape[0]; i++ {
		delta := err.at(i) * sigmoidDer(output.at(i))
		for j := 0; j < w.shape[1]; j++ {
			errout.add(delta*w.at(j, i), j)
			w.sub(lr*delta*input.at(j), j, i)
			// w[i][j] = w[i][j] - lr*delta*input[j]

		}
		b.sub(lr*delta*1, i)
		// b[i] = b[i] - lr*delta*1
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

func loss(arr Tensor, target Tensor) Tensor {
	resArr := Tensor{}
	resArr.zero(arr.shape[0])
	for i := 0; i < arr.shape[0]; i++ {
		temp := target.at(i) - arr.at(i)
		resArr.set(-temp, i)
	}
	return resArr
}

func totalLoss(arr Tensor, target Tensor) float32 {
	result := float32(0)
	for i := 0; i < arr.shape[0]; i++ {
		temp := (target.at(i) - arr.at(i))
		result += 0.5 * temp * temp
		// resArr.set(-temp, i)
	}
	return result
}

// func sumArr(arr Tensor) float32 {
// 	result := float32(0.0)
// 	for i := 0; i < arr.shape[0]; i++ {
// 		result += arr[i]
// 	}
// 	return result
// }

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
