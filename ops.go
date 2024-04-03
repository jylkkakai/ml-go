package main

import (
	// "fmt"
	"log"
	"math"
	"math/rand"
)

type DenseCReturn struct {
	ret float32
	i   int
}
type DenseBPCReturn struct {
	e []float32
	w []float32
	b float32
	i int
	j int
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
	c := make(chan DenseCReturn, w.shape[1])
	for i := 0; i < w.shape[1]; i++ {
		go func(in, w, b Tensor, activ activation, i int, ch chan DenseCReturn) {
			ret := DenseCReturn{}
			for j := 0; j < w.shape[0]; j++ {
				ret.ret += w.at(j, i) * in.at(j)
			}
			ret.i = i
			c <- ret
		}(in, w, b, activ, i, c)
	}
	for i := 0; i < w.shape[1]; i++ {
		temp := <-c
		out.set(activ(temp.ret+b.at(temp.i)), temp.i)
	}
	return out
}

func denseBP(w Tensor, b Tensor, err Tensor, input Tensor, output Tensor, lr float32, act string) Tensor {

	var activ activation
	switch act {
	case "relu":
		activ = reluDer
	case "sigmoid":
		activ = sigmoidDer
	default:
		activ = pass
	}
	errout := Tensor{}
	errout.zero(input.shape[0])
	c := make(chan DenseBPCReturn, w.shape[1])
	for i := 0; i < w.shape[1]; i++ {
		go func(i int) {
			delta := err.at(i) * activ(output.at(i))
			ret := DenseBPCReturn{}
			ret.e = make([]float32, errout.shape[0])
			ret.w = make([]float32, w.shape[0])
			for j := 0; j < w.shape[0]; j++ {
				ret.e[j] = delta * w.at(j, i)
				// errout.add(delta*w.at(j, i), j)
				ret.w[j] = lr * delta * input.at(j)
				// w.sub(lr*delta*input.at(j), j, i)

			}
			ret.i = i
			// ret.j = j
			ret.b = lr * delta * 1
			b.sub(lr*delta*1, i)
			c <- ret
		}(i)
	}
	for i := 0; i < w.shape[1]; i++ {
		temp := <-c
		// go func(i int) {
		for j := 0; j < w.shape[0]; j++ {
			errout.add(temp.e[j], j)
			w.sub(temp.w[j], j, temp.i)
			// b.sub(temp.b, temp.i)
			// out.set(activ(temp.ret+b.at(temp.i)), temp.i)
		}
		// }(i)
	}
	return errout
}

// 1D Tensor
func softmax(t Tensor) Tensor {

	ret := Tensor{}
	ret.zero(t.shape[0])
	// sum := t.sum()
	sum := float32(0)
	max := t.at(t.argmax())

	xval := Tensor{}
	xval.zero(t.shape[0])
	for i := 0; i < len(t.arr); i++ {
		xval.set(t.arr[i]-max, i)
	}
	for i := 0; i < len(t.arr); i++ {
		sum += float32(math.Exp(float64(xval.arr[i])))
	}
	for i := 0; i < t.shape[0]; i++ {
		temp := math.Exp(float64(xval.at(i)))
		ret.set(float32(temp)/sum, i)
	}
	return ret
}

func shuffleTrainingData(in, classes Tensor) {

	for i := classes.shape[0]; i > 0; i-- {
		idx := rand.Intn(i)
		tempi := in.deepSlice(idx)
		tempc := classes.at(idx)
		in.setSlice(in.slice(i-1), idx)
		classes.set(classes.at(i-1), idx)
		in.setSlice(tempi, i-1)
		classes.set(tempc, i-1)
	}
}

func loss(arr Tensor, target Tensor) Tensor {
	resArr := Tensor{}
	resArr.zero(arr.shape[0])
	for i := 0; i < arr.shape[0]; i++ {
		// temp := (arr.at(i) - target.at(i))
		temp := float32(2) / float32(arr.shape[0]) * (arr.at(i) - target.at(i))
		resArr.set(temp, i)
	}
	return resArr
}

func totalLoss(arr Tensor, target Tensor) float32 {
	result := float32(0)
	for i := 0; i < arr.shape[0]; i++ {
		temp := (target.at(i) - arr.at(i))
		result += temp * temp
		// resArr.set(-temp, i)
	}
	return result / float32(arr.shape[0])
}

func normalize(t Tensor) Tensor {

	ret := Tensor{}
	if len(t.shape) == 2 {
		ret.zero(t.shape[0], t.shape[1])
	} else if len(t.shape) == 3 {
		ret.zero(t.shape[0], t.shape[1], t.shape[2])
	} else {
		log.Fatalln("Only 2D or 3D tensors can be normalized.")
	}
	for i := 0; i < len(t.arr); i++ {
		ret.arr[i] = (t.arr[i] - float32(128)) / float32(128)
	}
	return ret
}

// Reverse normalize
func toRGB(t Tensor) Tensor {

	ret := Tensor{}
	if len(t.shape) == 2 {
		ret.zero(t.shape[0], t.shape[1])
	} else if len(t.shape) == 3 {
		ret.zero(t.shape[0], t.shape[1], t.shape[2])
	} else {
		log.Fatalln("Only 2D or 3D tensors can be normalized.")
	}
	for i := 0; i < len(t.arr); i++ {
		ret.arr[i] = float32(int(t.arr[i]*float32(128) + float32(128)))
	}
	return ret
}
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

func reluDer(x float32) float32 {
	if x <= 0 {
		return 0
	} else {
		return 1
	}
}
func activate(x, b float32, fn activation) float32 {
	raw := x + b
	return fn(raw)
}
