package main

import (
	"fmt"
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

func testTraining() {

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

	target.shape = target.shape[1:2]
	outLoss := loss(lout4, target)
	totLoss := totalLoss(lout4, target)
	l4loss := denseBP(w4, b4, outLoss, lout3, lout4, float32(0.5))
	// fmt.Println(b2)
	// fmt.Println(l4loss)
	// fmt.Println(lout3)
	l3loss := denseBP(w3, b3, l4loss, lout2, lout3, float32(0.5))
	l2loss := denseBP(w2, b2, l3loss, lout1, lout2, float32(0.5))
	l1loss := denseBP(w1, b1, l2loss, lout0, lout1, float32(0.5))
	l0loss := denseBP(w0, b0, l1loss, din, lout0, float32(0.5))
	_ = l0loss
}
