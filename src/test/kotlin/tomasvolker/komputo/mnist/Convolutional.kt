package tomasvolker.komputo.mnist

import tomasvolker.komputo.builder.*
import tomasvolker.komputo.dataset.mapLabels
import tomasvolker.komputo.dsl.RELU
import tomasvolker.komputo.functions.softmax
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.functions.argmax
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.primitives.indicator

fun main() {

    val (trainDataset, testDataset) = loadMnistDataset()

    println("train dataset size: ${trainDataset.size}")
    println("test dataset size: ${testDataset.size}")

    val model = trainableModel {

        sequential(inputShape = I[28, 28]) {
            conv2d(
                filterCount = 32,
                kernelSize = I[3, 3],
                activation = RELU
            )
            conv2d(
                filterCount = 64,
                kernelSize = I[3, 3],
                activation = RELU
            )
            maxPool2D(windowSize = I[2, 2])
            flatten()
            dropout(0.25)
            dense(128, activation = RELU)
            dropout(0.5)
            dense(10)
        }

        training {
            loss = crossEntropyWithLogits
            optimizer = Adagrad()
        }

    }

    session(model) {

        train {

            dataset = trainDataset.mapLabels { it.toOneHot(10) }

            epochs = 5
            batchSize = 128

            verbose()

        }


        fun classify(image: DoubleArray2D): DoubleArray1D =
            softmax(evaluate(image).first().as2D()[0, All])

        val testAccuracy = testDataset.map {
            (classify(it.data).argmax() == it.label).indicator()
        }.average()

        println("Test accuracy: %.2f%%".format(testAccuracy * 100))

    }

}