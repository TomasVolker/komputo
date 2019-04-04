package tomasvolker.komputo.mnist

import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.builder.*
import tomasvolker.komputo.dataset.labelTo
import tomasvolker.komputo.dataset.mapLabels
import tomasvolker.komputo.dsl.RELU
import tomasvolker.komputo.dsl.SIGMOID
import tomasvolker.komputo.functions.softmax
import tomasvolker.numeriko.core.dsl.D
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.functions.argmax
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.interfaces.array2d.generic.forEachIndex
import tomasvolker.numeriko.core.interfaces.array2d.generic.get
import tomasvolker.numeriko.core.interfaces.factory.array2D
import tomasvolker.numeriko.core.interfaces.factory.doubleArrayND
import tomasvolker.numeriko.core.performance.forEach

fun main() {

    val (trainDataset, testDataset) = loadMnistDataset().let {
        it.first.filter { it.label == 8 } to it.second
    }

    println("train dataset size: ${trainDataset.size}")
    println("test dataset size: ${testDataset.size}")

    lateinit var encoding: TFOperand

    val model = trainableModel {

        sequential(inputShape = I[28, 28]) {
            flatten()
            dense(512, activation = RELU)
            encoding = dense(2, activation = SIGMOID)
            dense(28 * 28, activation = RELU)
            reshape(I[28, 28])
        }

        training {
            loss = metric { output, target ->
                reduceMean(square(target - output), I[0, 1, 2])
            }
            optimizer = Adagrad()
        }

    }

    session(model) {

        train {

            dataset = trainDataset.map { it.data labelTo it.data }

            epochs = 1000
            batchSize = 128

            verbose()

        }


        fun decode(vector: DoubleArray1D) =
            evaluate(
                operand = model.outputList.first(),
                feed = mapOf(encoding to vector.higherRank(axis = 0))
            ).arrayAlongAxis(axis = 0, index = 0).as2D()

        val imageMatrix = array2D(10, 10) { i0, i1 ->
            val x = i0.toDouble() / 10
            val y = i1.toDouble() / 10
            decode(D[x, y])
        }

        showImageMatrix(imageMatrix)

    }

}