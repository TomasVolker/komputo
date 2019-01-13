package tomasvolker.komputo.mnist

import tomasvolker.komputo.builder.*
import tomasvolker.komputo.dataset.LabeledDataset
import tomasvolker.komputo.dataset.mapLabels
import tomasvolker.komputo.dsl.RELU
import tomasvolker.komputo.functions.softmax
import tomasvolker.komputo.performance.argmax
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.factory.doubleArray1D
import tomasvolker.numeriko.core.primitives.indicative
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D

fun main() {

    val (trainDataset, testDataset) = loadMnistDataset()

    println("train dataset size: ${trainDataset.size}")
    println("test dataset size: ${testDataset.size}")

    val model = trainableModel {

        sequential(inputShape = I[28, 28]) {
            flatten()
            dense(512, activation = RELU)
            dense(10)
        }

        training {
            loss = crossEntropyWithLogits
            optimizer = Adagrad()
        }

    }

    session(model, initialize = false) {
/*
        train {

            dataset = trainDataset.mapLabels { it.toOneHot(10) }

            epochs = 5
            batchSize = 128

            verbose()

        }
*/
        restore("test_save.pb")

        fun classify(image: DoubleArray2D): DoubleArray1D =
            softmax(evaluate(image).first().as2D()[0, All])

        val testAccuracy = testDataset.map {
            (classify(it.data).argmax() == it.label).indicative()
        }.average()

        println("Test accuracy: %.2f%%".format(testAccuracy * 100))

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



typealias MnistDataset = LabeledDataset<DoubleArray2D, Int>

fun loadMnistDataset(): Pair<MnistDataset, MnistDataset> {
    val trainDataset = Mnist.loadDataset(
        imagesPath = "data/train-images-idx3-ubyte",
        labelsPath = "data/train-labels-idx1-ubyte"
    )

    val testDataset = Mnist.loadDataset(
        imagesPath = "data/t10k-images-idx3-ubyte",
        labelsPath = "data/t10k-labels-idx1-ubyte"
    )
    return trainDataset to testDataset
}