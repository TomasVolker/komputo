package tomasvolker.komputo.mnist

import org.tensorflow.Operand
import tomasvolker.komputo.dsl.builder.*
import tomasvolker.kyplot.dsl.line
import tomasvolker.kyplot.dsl.showPlot
import tomasvolker.kyplot.dsl.xAxis
import tomasvolker.kyplot.dsl.yAxis
import tomasvolker.kyplot.model.Axis
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.factory.doubleArray1D
import tomasvolker.numeriko.core.operations.stack
import tomasvolker.numeriko.core.primitives.indicative
import tomasvolker.performance.reduceArgmax
import tomasvolker.performance.stack
import tomasvolker.tensorflow.dsl.*
import kotlin.system.measureTimeMillis

var lastLine = ""

fun renderLine(line: String) {
    clearLine()
    print(line)
    lastLine = line

}

fun clearLine() {
    print("\b".repeat(lastLine.length))
    lastLine = ""
}

fun nextLine() {
    println()
    lastLine = ""
}

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

    val model = trainableModel {

        val input = input<Float>(shape = I[dynamic, 28, 28])

        val model = sequential(input) {

            /*
            flatten()
            dense(512, activation = ops::relu)
            dense(10, activation = ops::identity)
            */

            reshape(I[28, 28, 1])
            conv2d(
                kernelSize = I[3, 3],
                filterCount = 32,
                activation = ops::relu
            )
            conv2d(
                kernelSize = I[3, 3],
                filterCount = 64,
                activation = ops::relu
            )
            maxPool2D(
                windowSize = I[2, 2],
                strides = I[1, 1]
            )
            flatten()
            dense(128, activation = ops::sigmoid)
            dense(10, activation = ops::identity)


        }

        output<Float>(model)

        loss = ops.softmaxCrossEntropyWithLogits(
            output as Operand<Float>,
            target as Operand<Float>
        ).loss().asOutput()

        trainingAlgorithm = Adagrad()
    }

    val epochLosses = mutableListOf<Double>()

    trainSession(model) {

        model.initialize()

        val millis = measureTimeMillis {

            val batchSize = 128

            repeat(5) { epoch ->

                println("Epoch $epoch")

                val batchList = trainDataset.shuffled().chunked(batchSize)

                val losses = batchList.withIndex().map { (i, batch) ->

                    val inputTensor = stack(batch.map { it.data })
                    val output = stack(batch.map { it.label.toOneHot(10) })

                    model.fit(inputTensor, output).first().getValue().also {
                        renderLine("Batch loss ${i * batchSize}/${trainDataset.size}: $it")
                    }
                }

                clearLine()

                println("Mean loss: ${losses.average().also { epochLosses.add(it) }}")

                val testBatch = List(10) { trainDataset.random() }

                val inputTensor = stack(testBatch.map { it.data })
                val predictions = model(inputTensor).first()

                val predictedLabels = predictions.as2D().reduceArgmax(1)
                testBatch.zip(predictedLabels) { data, predictedLabel ->
                    "${data.label} -> $predictedLabel"
                }.joinToString(separator = " | ").also { println(it) }

            }

        }

        println("seconds: ${millis / 1000.0}")

        repeat(5) {

            val (image, label) = testDataset.random()

            val predicted = model(image.higherRank()).first().as2D().reduceArgmax(1)

            print(Mnist.renderToString(image))
            println("predicted: $predicted, label: $label")
        }

    }

    showPlot {

        line {
            x = epochLosses.indices
            y = epochLosses
            label = "Epoch losses"
        }

        yAxis {
            label = "Loss"
            scale = Axis.Scale.LOGARITHMIC
        }

        xAxis {
            label = "Epoch"
        }

    }

}

fun Int.toOneHot(size: Int): DoubleArray1D =
    doubleArray1D(size) { i -> (i == this).indicative() }
