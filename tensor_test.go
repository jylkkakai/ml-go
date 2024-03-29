package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
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

	for i := 0; i < 12; i++ {

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
