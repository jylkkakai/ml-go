package main

import (
	"fmt"
	"github.com/kshedden/gonpy"
	"log"
	"math/rand"
	"time"
)

type Tensor struct {
	shape []int
	arr   []float32
}

// Prints only 2D tensor
func (t *Tensor) print() {

	fmt.Printf("\n")
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			fmt.Printf("%3v ", t.at(i, j))
		}
		fmt.Printf("\n")

	}
	fmt.Printf("\n")
}
func (t *Tensor) String() string {

	ret := ""
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			ret += fmt.Sprintf("%v ", t.at(i, j))
		}
		ret += "\n"
	}
	return ret
}

func (t *Tensor) getIndex(coord []int) int {
	ii := 0
	tempm := 1
	for i := len(t.shape) - 1; i >= 0; i-- {
		temp := coord[i]

		if i < len(t.shape)-1 {
			tempm *= t.shape[i+1]
			temp = tempm * coord[i]
		}
		ii += temp

	}
	return ii
}
func (t *Tensor) at(coord ...int) float32 {
	return t.arr[t.getIndex(coord)]
}
func (t *Tensor) set(x float32, coord ...int) {
	t.arr[t.getIndex(coord)] = x
}

func (t *Tensor) add(x float32, coord ...int) {
	t.arr[t.getIndex(coord)] += x
}

func (t *Tensor) sub(x float32, coord ...int) {
	t.arr[t.getIndex(coord)] -= x
}

// For now only slices mnist data [28, 28]
func (t *Tensor) slice(x int) Tensor {
	w := t.shape[1]
	h := t.shape[2]
	ret := Tensor{}
	ret.zero(w, h)
	ret.arr = t.arr[x*w*h : x*w*h+w*h]
	return ret
}

func (t *Tensor) readNpy(f string) {

	r, err := gonpy.NewFileReader(f)
	if err != nil {
		log.Fatalln(err)
	}
	data, err1 := r.GetFloat32()
	if err1 != nil {
		log.Println(err)
	}
	t.shape = r.Shape
	t.arr = data
}
func (t *Tensor) zero(s ...int) {
	len := 1
	for _, v := range s {
		len *= v
	}
	t.arr = make([]float32, len)
	t.shape = s
}

func (t *Tensor) random(s ...int) {
	len := 1
	for _, v := range s {
		len *= v
	}
	t.arr = make([]float32, len)
	t.shape = s
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < len; i++ {
		t.arr[i] = r.Float32()*2 - 1
	}
}

func (t *Tensor) shuffle(s ...int) {

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	r.Shuffle(len(t.arr), func(i, j int) { t.arr[i], t.arr[j] = t.arr[j], t.arr[i] })
}
