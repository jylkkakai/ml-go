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

func TestDenseBP(t *testing.T) {

	w0 := Tensor{}
	w0.readNpy("test_data/test_denseBP_w0.npy")
	w1 := Tensor{}
	w1.readNpy("test_data/test_denseBP_w1.npy")
	w2 := Tensor{}
	w2.readNpy("test_data/test_denseBP_w2.npy")
	w3 := Tensor{}
	w3.readNpy("test_data/test_denseBP_w3.npy")
	w4 := Tensor{}
	w4.readNpy("test_data/test_denseBP_w4.npy")
	fw0 := Tensor{}
	fw0.readNpy("test_data/test_denseBP_fw0.npy")
	fw1 := Tensor{}
	fw1.readNpy("test_data/test_denseBP_fw1.npy")
	fw2 := Tensor{}
	fw2.readNpy("test_data/test_denseBP_fw2.npy")
	fw3 := Tensor{}
	fw3.readNpy("test_data/test_denseBP_fw3.npy")
	fw4 := Tensor{}
	fw4.readNpy("test_data/test_denseBP_fw4.npy")
	b0 := Tensor{}
	b0.readNpy("test_data/test_denseBP_b0.npy")
	b1 := Tensor{}
	b1.readNpy("test_data/test_denseBP_b1.npy")
	b2 := Tensor{}
	b2.readNpy("test_data/test_denseBP_b2.npy")
	b3 := Tensor{}
	b3.readNpy("test_data/test_denseBP_b3.npy")
	b4 := Tensor{}
	b4.readNpy("test_data/test_denseBP_b4.npy")
	fb0 := Tensor{}
	fb0.readNpy("test_data/test_denseBP_fb0.npy")
	fb1 := Tensor{}
	fb1.readNpy("test_data/test_denseBP_fb1.npy")
	fb2 := Tensor{}
	fb2.readNpy("test_data/test_denseBP_fb2.npy")
	fb3 := Tensor{}
	fb3.readNpy("test_data/test_denseBP_fb3.npy")
	fb4 := Tensor{}
	fb4.readNpy("test_data/test_denseBP_fb4.npy")

	gloss := Tensor{}
	gloss.readNpy("test_data/test_denseBP_loss.npy")
	din := Tensor{}
	din.readNpy("test_data/test_denseBP_din.npy")
	dout := Tensor{}
	dout.readNpy("test_data/test_denseBP_dout.npy")
	target := Tensor{}
	target.readNpy("test_data/test_denseBP_target.npy")

	lout0 := dense(din, w0, b0, "sigmoid")
	lout1 := dense(lout0, w1, b1, "sigmoid")
	lout2 := dense(lout1, w2, b2, "sigmoid")
	lout3 := dense(lout2, w3, b3, "sigmoid")
	lout4 := dense(lout3, w4, b4, "sigmoid")

	// fmt.Println("lout0:", lout0)
	// fmt.Println("lout1:", lout1)
	// fmt.Println("lout2", lout2)
	// fmt.Println("lout3:", lout3)
	// fmt.Println("lout4:", lout4)
	// fmt.Println("dout:", dout)
	target.shape = target.shape[1:2]
	// fmt.Println(target)
	outLoss := loss(lout4, target)
	fmt.Println("outloss:", outLoss)
	totLoss := totalLoss(lout4, target)
	// fmt.Println(totLoss)
	// fmt.Println(gloss)
	fmt.Println(w4)
	l4loss := denseBP(w4, b4, outLoss, lout3, lout4, float32(0.5))
	// fmt.Println(b2)
	// fmt.Println(l4loss)
	// fmt.Println(lout3)
	l3loss := denseBP(w3, b3, l4loss, lout2, lout3, float32(0.5))
	l2loss := denseBP(w2, b2, l3loss, lout1, lout2, float32(0.5))
	l1loss := denseBP(w1, b1, l2loss, lout0, lout1, float32(0.5))
	l0loss := denseBP(w0, b0, l1loss, din, lout0, float32(0.5))
	_ = l0loss
	// fmt.Println(gloss)
	// fmt.Println(totLoss)

	// Change values to see that test fails
	// if i == 8 {
	// 	w.arr[0] = float32(0.0)
	// 	dout.arr[1] = float32(1.0)
	// }

	t.Run("Compare FP output", func(t *testing.T) {
		for i := 0; i < dout.shape[1]; i++ {
			eps := float32(0.00001)
			if lout4.at(i)+eps < dout.at(0, i) || lout4.at(i)-eps > dout.at(0, i) {
				t.Errorf("Output: %f != %f in %d is incorrect!", lout4.at(i), dout.at(0, i), i)
			}
		}
	})
	t.Run("Compare total loss", func(t *testing.T) {
		eps := float32(0.00001)
		if totLoss+eps < gloss.at(0, 0) || totLoss-eps > gloss.at(0, 0) {
			t.Errorf("Output: %f != %f is incorrect!", totLoss-eps, gloss.at(0, 0))
		}
	})
	fmt.Println(w4.shape)
	t.Run("Compare w4", func(t *testing.T) {
		compareTensor(t, w4, fw4)
	})
	// fmt.Println(b4)
	// fmt.Println(fb4)
	t.Run("Compare b4", func(t *testing.T) {
		compareTensor(t, b4, fb4)
	})
	t.Run("Compare w3", func(t *testing.T) {
		compareTensor(t, w3, fw3)
	})
	t.Run("Compare b3", func(t *testing.T) {
		compareTensor(t, b3, fb3)
	})
	t.Run("Compare w2", func(t *testing.T) {
		compareTensor(t, w2, fw2)
	})
	// fmt.Println(b4)
	// fmt.Println(fb4)
	t.Run("Compare b2", func(t *testing.T) {
		compareTensor(t, b2, fb2)
	})
	t.Run("Compare w1", func(t *testing.T) {
		compareTensor(t, w1, fw1)
	})
	// Test fail
	// b1.set(0.1, 1)
	t.Run("Compare b1", func(t *testing.T) {
		compareTensor(t, b1, fb1)
	})
	t.Run("Compare w0", func(t *testing.T) {
		compareTensor(t, w0, fw0)
	})
	// fmt.Println(b4)
	// fmt.Println(fb4)
	t.Run("Compare b0", func(t *testing.T) {
		compareTensor(t, b0, fb0)
	})
}
func compareTensor(t *testing.T, x, y Tensor) {
	eps := float32(0.00001)
	for i := 0; i < len(x.arr); i++ {
		// fmt.Printf("Output: %f != %f \tin %d, %d is incorrect!\n", w4.at(i, j), fw4.at(i, j), i, j)
		if x.arr[i]+eps < y.arr[i] || x.arr[i]-eps > y.arr[i] {
			t.Errorf("Output: %f != %f \tin %d is incorrect!", x.arr[i], y.arr[i], i)
		}
		// for j := 0; j < x.shape[1]; j++ {
		// 	// fmt.Printf("Output: %f != %f \tin %d, %d is incorrect!\n", w4.at(i, j), fw4.at(i, j), i, j)
		// 	if x.at(i, j)+eps < y.at(i, j) || x.at(i, j)-eps > y.at(i, j) {
		// 		t.Errorf("Output: %f != %f \tin %d, %d is incorrect!", x.at(i, j), y.at(i, j), i, j)
		// 	}
		// }
	}
}
