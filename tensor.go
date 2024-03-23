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

func (t *Tensor) at(coord ...int) float32 {
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
	return t.arr[ii]
}
func (t *Tensor) set(x float32, coord ...int) {
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
	t.arr[ii] = x
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
