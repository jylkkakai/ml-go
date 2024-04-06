package main

import (
	"fmt"
	"github.com/kshedden/gonpy"
	"log"
	"math/rand"
	// "sync"
	"time"
)

type Tensor struct {
	shape []int
	arr   []float32
	// mu    sync.Mutex
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
	// t.mu.Lock()
	// defer t.mu.Unlock()
	t.arr[t.getIndex(coord)] = x
}

// 2D slice
func (t *Tensor) setSlice(x *Tensor, idx int) {
	// t.mu.Lock()
	// defer t.mu.Unlock()
	temp := []int{idx, 0, 0}
	size := x.shape[0] * x.shape[1]
	index := t.getIndex(temp)
	// fmt.Println(idx, size, index)
	for i := 0; i < size; i++ {
		t.arr[index+i] = x.arr[i]
	}
}
func (t *Tensor) add(x float32, coord ...int) {
	// t.mu.Lock()
	// defer t.mu.Unlock()
	t.arr[t.getIndex(coord)] += x
}

func (t *Tensor) sub(x float32, coord ...int) {
	// t.mu.Lock()
	// defer t.mu.Unlock()
	t.arr[t.getIndex(coord)] -= x
}

func (t *Tensor) flatten() {
	t.shape[0] = t.shape[0] * t.shape[1]
	t.shape = t.shape[0:1]
}

// For now only slices mnist data [28, 28]
func (t *Tensor) slice(x int) *Tensor {
	w := t.shape[1]
	h := t.shape[2]
	ret := Tensor{}
	ret.zero(w, h)
	ret.arr = t.arr[x*w*h : x*w*h+w*h]
	return &ret
}

// For now only slices mnist data [28, 28]
func (t *Tensor) deepSlice(x int) *Tensor {
	w := t.shape[1]
	h := t.shape[2]
	ret := Tensor{}
	ret.shape = append([]int(nil), t.shape[1:3]...)
	ret.arr = append([]float32(nil), t.arr[x*w*h:x*w*h+w*h]...)
	return &ret
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
		t.arr[i] = (r.Float32()*2 - 1) / float32(10)
	}
}

func (t *Tensor) sum() float32 {

	sum := float32(0)
	for i := 0; i < len(t.arr); i++ {

		sum += t.arr[i]
	}
	return sum
}

func (t *Tensor) shuffle() {

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	r.Shuffle(len(t.arr), func(i, j int) { t.arr[i], t.arr[j] = t.arr[j], t.arr[i] })
}

func (t *Tensor) argmax() int {

	max := 0
	for i := 1; i < len(t.arr); i++ {
		if t.arr[max] < t.arr[i] {
			max = i
		}
	}
	return max
}
