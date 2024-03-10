package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

type tensor [][][][]float32

func main() {

	const h = 20
	const w = 30
	const c = 32

	inStrArr := readFile("test/in.txt")
	wgtStrArr := readFile("test/wgt.txt")
	outStrArr := readFile("test/out.txt")

	inArr := strArrToTensor(inStrArr, 1, h, w, c)
	wgtArr := strArrToTensor(wgtStrArr, 3, 3, 32, 32)
	outArr := strArrToTensor(outStrArr, 1, h, w, c)
	fmt.Println(inArr[0][0][0])
	fmt.Println(wgtArr[0][0][0])
	fmt.Println(outArr[0][0][0])
}

func dense(in []float32, wgt [][]float32, relu bool) []float32 {

	out := make([]float32, len(wgt))
	for i, k := range wgt {
		for j := range in {
			if relu {
				out[i] = max(0, in[j]*k[j])
			} else {
				out[i] = in[j] * k[j]
			}
		}
	}
	return nil
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

func strArrToFlat(s []string, x, y int) []float32 {
	if len(s) < x*y {
		log.Fatalf("func strArrToTensor: Input string length %d smaller than array size %d.\n", len(s), x*y)
	}
	return nil
}

func strArrToTensor(s []string, b, h, w, c int) tensor {

	if len(s) < b*h*w*c {
		log.Fatalf("func strArrToTensor: Input string length %d smaller than array size %d.\n", len(s), b*h*w*c)
	}
	arr := make([][][][]float32, b)
	si := 0
	for i := 0; i < b; i++ {
		arr[i] = make([][][]float32, h)
		for j := 0; j < h; j++ {
			arr[i][j] = make([][]float32, w)
			for k := 0; k < w; k++ {
				arr[i][j][k] = make([]float32, c)
				for l := 0; l < c; l++ {
					if len(s[si]) > 0 {
						temp, err := strconv.ParseFloat(s[si], 32)
						if err != nil {
							log.Fatalln(err, i, j, k, l, temp, s[si], len(s[si]))
						}
						arr[i][j][k][l] = float32(temp)
					}
					si++
				}
			}
		}
	}
	return arr
}
