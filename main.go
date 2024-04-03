package main

import (
	"fmt"
	"time"
)

func main() {

	// simpleNet()
	// testTraining()
	mnist()
}

func mnist() {

	fmt.Println("Dense fp and bp kernels threaded, slices, all training data shuffled, channels buffered.")
	epochs := 6
	batchSize := 10000
	lr := float32(0.01)
	fmt.Printf("lr = %f, nn = (128, 128, 10)\n", lr)
	xtrainRGB := Tensor{}
	xtrainRGB.readNpy("test_data/mnist_x_train.npy")
	xtrain := normalize(xtrainRGB)
	ytrain := Tensor{}
	ytrain.readNpy("test_data/mnist_y_train.npy")
	xtestRGB := Tensor{}
	xtestRGB.readNpy("test_data/mnist_x_test.npy")
	xtest := normalize(xtestRGB)
	ytest := Tensor{}
	ytest.readNpy("test_data/mnist_y_test.npy")
	target := Tensor{}
	din := Tensor{}
	numOfCor := 0
	cumLoss := float32(0)
	totLoss := float32(0)

	w0 := Tensor{}
	w0.random(28*28, 128)
	b0 := Tensor{}
	b0.zero(128)
	w1 := Tensor{}
	w1.random(128, 128)
	b1 := Tensor{}
	b1.zero(128)
	w2 := Tensor{}
	w2.random(128, 10)
	b2 := Tensor{}
	b2.zero(10)

	totStart := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {

		shuffleTrainingData(xtrain, ytrain)
		epochStart := time.Now()
		fmt.Println("---------------------")
		fmt.Println("Training...")
		start := time.Now()
		for i := 0; i < ytrain.shape[0]; i++ { // ytrain.shape[0]

			din = xtrain.slice(i)
			din.flatten()
			lout0 := dense(din, w0, b0, "relu")
			lout1 := dense(lout0, w1, b1, "relu")
			lout2 := dense(lout1, w2, b2, "")
			sfout := softmax(lout2)

			target.zero(10)
			target.set(float32(1), int(ytrain.at(i)))
			outLoss := loss(sfout, target)
			totLoss = totalLoss(sfout, target)
			cumLoss += totLoss
			maxArg := sfout.argmax()
			if maxArg == int(ytrain.at(i)) {
				numOfCor += 1
			}

			l2loss := denseBP(w2, b2, outLoss, lout1, sfout, lr, "relu")
			l1loss := denseBP(w1, b1, l2loss, lout0, lout1, lr, "relu")
			_ = denseBP(w0, b0, l1loss, din, lout0, lr, "")

			if (i+1)%batchSize == 0 {
				t := time.Now()
				elapsed := t.Sub(start)
				fmt.Printf("Epoch: %d \t%d/%d\tElapsed time: %v \tAvg loss: %f \t accuracy: %f\t\n", epoch+1, i+1, ytrain.shape[0], elapsed, cumLoss/float32(batchSize), float32(numOfCor)/float32(batchSize))
				start = time.Now()
				numOfCor = 0
				cumLoss = float32(0)

			}
		}
		t := time.Now()
		fmt.Printf("Epoch elapsed time: %v\n", t.Sub(epochStart))
		start = time.Now()
		fmt.Println("---------------------")
		fmt.Println("Running validation...")
		for i := 0; i < ytest.shape[0]; i++ { // ytrain.shape[0]

			din = xtest.slice(i)
			din.flatten()
			lout0 := dense(din, w0, b0, "relu")
			lout1 := dense(lout0, w1, b1, "relu")
			lout2 := dense(lout1, w2, b2, "")
			sfout := softmax(lout2)

			target.zero(10)
			target.set(float32(1), int(ytest.at(i)))
			totLoss = totalLoss(sfout, target)
			cumLoss += totLoss
			maxArg := sfout.argmax()

			if maxArg == int(ytest.at(i)) {
				numOfCor += 1
			}

			if (i+1)%ytest.shape[0] == 0 {
				t := time.Now()
				elapsed := t.Sub(start)
				fmt.Printf("Epoch: %d \t%d/%d\tElapsed time: %v \tAvg loss: %f \t accuracy: %f\t\n", epoch+1, i+1, ytest.shape[0], elapsed, cumLoss/float32(ytest.shape[0]), float32(numOfCor)/float32(ytest.shape[0]))
				// finalLoss = totLoss
				start = time.Now()

				numOfCor = 0
				cumLoss = float32(0)

			}
		}
		// lr = lr * float32(0.5)
	}
	totElapsed := time.Since(totStart)
	fmt.Printf("Total training time: %v \n", totElapsed)

	for i := 0; i < 10; i++ {
		offset := 30000
		din = xtrain.slice(i + offset)
		din.flatten()
		lout0 := dense(din, w0, b0, "relu")
		lout1 := dense(lout0, w1, b1, "relu")
		lout2 := dense(lout1, w2, b2, "")
		sfout := softmax(lout2)
		// lout0 := dense(din, w0, b0, "relu")
		// lout1 := dense(lout0, w1, b1, "")
		// sfout := softmax(lout1)

		cumLoss += totLoss
		maxArg := sfout.argmax()
		img := xtrain.slice(i + offset)
		img = toRGB(img)
		img.print()
		fmt.Println("Prediction:", maxArg)
	}
}

func testTraining() {

	epochs := 100
	batchSize := 1
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
	target.shape = target.shape[1:2]

	totStart := time.Now()
	finalLoss := float32(0.0)
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batchSize; batch++ {

			start := time.Now()
			lout0 := dense(din, w0, b0, "sigmoid")
			lout1 := dense(lout0, w1, b1, "sigmoid")
			lout2 := dense(lout1, w2, b2, "sigmoid")
			lout3 := dense(lout2, w3, b3, "sigmoid")
			lout4 := dense(lout3, w4, b4, "sigmoid")

			outLoss := loss(lout4, target)
			totLoss := totalLoss(lout4, target)
			l4loss := denseBP(w4, b4, outLoss, lout3, lout4, float32(0.5), "sigmoid")
			// fmt.Println(b2)
			// fmt.Println(l4loss)
			// fmt.Println(lout3)
			l3loss := denseBP(w3, b3, l4loss, lout2, lout3, float32(0.5), "sigmoid")
			l2loss := denseBP(w2, b2, l3loss, lout1, lout2, float32(0.5), "sigmoid")
			l1loss := denseBP(w1, b1, l2loss, lout0, lout1, float32(0.5), "sigmoid")
			_ = denseBP(w0, b0, l1loss, din, lout0, float32(0.5), "sigmoid")
			t := time.Now()
			elapsed := t.Sub(start)

			fmt.Printf("Epoch: %d \tElapsed time: %v \tTotal loss: %f\n", epoch+1, elapsed, totLoss)
			finalLoss = totLoss
		}
	}
	totElapsed := time.Since(totStart)
	fmt.Printf("Total training time: %v \tFinal loss: %f\n", totElapsed, finalLoss)

}

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
	l2Loss := denseBP(w2, b2, outLoss, dout1, dout2, float32(0.5), "sigmoid")
	fmt.Println(w2)
	fmt.Println(b2)
	fmt.Println(l2Loss)
	l1Loss := denseBP(w1, b1, l2Loss, in, dout1, float32(0.5), "sigmoid")
	fmt.Println(w1)
	fmt.Println(b1)
	fmt.Println(l1Loss)

}
