package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	// "reflect"
	"strconv"
	"strings"

	// "strings"
	"testing"
)

func toInt(s string) int {

	value, err := strconv.Atoi(s)
	if err != nil {
		log.Fatal(err)
	}
	return value
}
func arrToInt(s []string) []int {
	arr := make([]int, len(s))
	for i, c := range s {
		arr[i] = toInt(c)
	}
	return arr
}
func compare(t *testing.T, x, y float32) {
	if x != y {
		t.Errorf("Incorrect value, got: %f want: %f\n", x, y)
	}
}
func TestAt(test *testing.T) {

	for i := 0; i < 6; i++ {

		filename := fmt.Sprintf("test_data/test_at%d", i)

		test.Run(filename, func(test *testing.T) {

			readData, err := os.ReadFile(filename + ".golden")
			if err != nil {
				test.Fatal(err)
			}
			tensor := Tensor{}
			tensor.readNpy(filename + ".npy")
			// fmt.Println(tensor)
			data := strings.Split(strings.ReplaceAll(string(readData), " ", ""), "\n")

			for _, line := range data {

				if len(line) > 0 {
					sline := strings.Split(line, ";")
					gValue, err := strconv.ParseFloat(sline[1], 32)
					if err != nil {
						test.Fatal(err)
					}
					gIndex := arrToInt(strings.Split(strings.Split(strings.Split(sline[0], "[")[1], "]")[0], ","))
					// _ = gValue
					// fmt.Println(gIndex)

					switch len(gIndex) {
					case 1:
						// fmt.Println(tensor.at(gIndex[0]), float32(gValue))
						compare(test, tensor.at(gIndex[0]), float32(gValue))
					case 2:
						compare(test, tensor.at(gIndex[0], gIndex[1]), float32(gValue))
					case 3:
						// fmt.Println(tensor.at(gIndex[0]), float32(gValue))
						compare(test, tensor.at(gIndex[0], gIndex[1], gIndex[2]), float32(gValue))
					case 4:
						compare(test, tensor.at(gIndex[0], gIndex[1], gIndex[2], gIndex[3]), float32(gValue))
					case 5:
						compare(test, tensor.at(gIndex[0], gIndex[1], gIndex[2], gIndex[3], gIndex[4]), float32(gValue))
					case 6:
						compare(test, tensor.at(gIndex[0], gIndex[1], gIndex[2], gIndex[3], gIndex[4], gIndex[5]), float32(gValue))
					}

				}

			}
		})

	}

}

func TestSet(test *testing.T) {

	tensor := Tensor{}
	tensor.zero(2, 3, 4, 5, 6)

	for i := 0; i < 100; i++ {

		value := rand.Float32()
		j := rand.Intn(tensor.shape[0])
		k := rand.Intn(tensor.shape[1])
		l := rand.Intn(tensor.shape[2])
		m := rand.Intn(tensor.shape[3])
		n := rand.Intn(tensor.shape[4])
		tensor.set(value, j, k, l, m, n)
		compare(test, value, tensor.at(j, k, l, m, n))
	}

}

func TestDense(t *testing.T) {

	for i := 0; i < 10; i++ {
		filename := fmt.Sprintf("test_data/test_dense%d", i)
		t.Run(filename, func(t *testing.T) {
			w := Tensor{}
			w.readNpy(filename + "_w.npy")
			b := Tensor{}
			b.readNpy(filename + "_b.npy")
			din := Tensor{}
			din.readNpy(filename + "_din.npy")
			dout := Tensor{}
			dout.readNpy(filename + "_dout.npy")

			// Change values to see that test fails
			// if i == 8 {
			// 	w.arr[0] = float32(0.0)
			// 	dout.arr[1] = float32(1.0)
			// }

			out := dense(din, w, b, "sigmoid")
			//
			// if reflect.DeepEqual(out, dout) {
			// 	t.Errorf("Output is incorrect!")
			// }
			for i, v := range out.arr {
				eps := float32(0.00001)
				if v+eps < dout.at(i) || v-eps > dout.at(i) {
					t.Errorf("Output: %f != %f in %d is incorrect!", v, dout.at(i), i)
				}
			}
		})
	}
}
