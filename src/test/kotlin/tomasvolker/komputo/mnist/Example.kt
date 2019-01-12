package tomasvolker.komputo.mnist

import tomasvolker.komputo.builder.*
import tomasvolker.komputo.dataset.mapLabels
import tomasvolker.komputo.performance.argmax
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.factory.doubleArray1D
import tomasvolker.numeriko.core.primitives.indicative
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D

fun main() {

    val trainDataset = Mnist.loadDataset(
        imagesPath = "data/train-images-idx3-ubyte",
        labelsPath = "data/train-labels-idx1-ubyte"
    )

    val testDataset = Mnist.loadDataset(
        imagesPath = "data/t10k-images-idx3-ubyte",
        labelsPath = "data/t10k-labels-idx1-ubyte"
    )

    println("train dataset size: ${trainDataset.size}")
    println("test dataset size: ${testDataset.size}")


    val model = graphModel {

        val input = input(shape = I[dynamic, 28, 28])

        val logits = sequential(input) {

            reshape(I[28, 28, 1])
            conv2d(
                kernelSize = I[3, 3],
                filterCount = 32,
                activation = builder::relu
            )
            conv2d(
                kernelSize = I[3, 3],
                filterCount = 64,
                activation = builder::relu
            )
            maxPool2D(
                windowSize = I[2, 2],
                strides = I[1, 1]
            )
            flatten()
            dropout(0.25)
            dense(128,
                activation = builder::sigmoid
            )
            dense(10)

        }

        output(logits)
        output(softmax(logits))

    }


    session(model) {

        train {

            dataset = trainDataset.mapLabels { it.toOneHot(10) }

            epochs = 5
            batchSize = 128

            loss = ::crossEntropyWithLogits
            optimizer = Adagrad()

            verbose()

        }

        fun classify(image: DoubleArray2D): DoubleArray1D = model(image)[1].as2D()[0, All]

        val testAccuracy = testDataset.map {
            (classify(it.data).argmax() == it.label).indicative()
        }.average()

        println("Test accuracy: ${testAccuracy * 100}%")

    }

}

fun highestPredictionsString(probabilities: DoubleArray1D) =
    probabilities.toList()
        .withIndex()
        .sortedByDescending { it.value }
        .take(3)
        .joinToString(prefix = "{", postfix = "}") { "%d:%.2f%%".format(it.index, it.value*100) }


fun Int.toOneHot(size: Int): DoubleArray1D =
    doubleArray1D(size) { i -> (i == this).indicative() }



