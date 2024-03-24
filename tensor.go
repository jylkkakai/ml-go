package main

import (
	// "fmt"
	"github.com/kshedden/gonpy"
	"log"
)

type Tensor struct {
	shape []int
	arr   []float32
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
