package tomasvolker.komputo.mnist

import org.tensorflow.Operand
import tomasvolker.komputo.dsl.builder.*
import tomasvolker.kyplot.dsl.*
import tomasvolker.kyplot.model.Axis
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.unsafeGetView
import tomasvolker.numeriko.core.interfaces.factory.doubleArray1D
import tomasvolker.numeriko.core.operations.stack
import tomasvolker.numeriko.core.primitives.indicative
import tomasvolker.performance.reduceArgmax
import tomasvolker.performance.stack
import tomasvolker.tensorflow.dsl.*
import kotlin.system.measureTimeMillis


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

            reshape(I[28, 28, 1])

            conv2d(
                kernelSize = I[3, 3],
                filterCount = 16
            )

            conv2d(
                kernelSize = I[3, 3],
                filterCount = 1
            )

            reshape(I[28, 28])

        }

        output<Float>(model)

        loss = meanSquareError(output as Operand<Float>, target as Operand<Float>)

        trainingAlgorithm = Adagrad()
    }

    val epochLosses = mutableListOf<Double>()

    trainSession(model) {

        model.initialize()

        val millis = measureTimeMillis {

            val batchSize = 32

            repeat(10) { epoch ->

                println("Epoch $epoch")

                val batchList = trainDataset.shuffled().chunked(batchSize)

                val losses = batchList.map { batch ->

                    val inputTensor = stack(batch.map { it.data })
                    //val output = stack(batch.map { it.label.toOneHot(10) })

                    model.fit(inputTensor, inputTensor).first().getValue()
                }

                val testBatch = List(3) { trainDataset.random() }

                val inputTensor = stack(testBatch.map { it.data })
                val predictions = model(inputTensor).first()

                val predictionList = List(predictions.shape(0)) { i ->
                    predictions.unsafeGetView(i, All, All).as2D()
                }

                testBatch.zip(predictionList) { data, output ->
                    println(Mnist.renderToString(data.data))
                    println(Mnist.renderToString(output))
                }

                println("Mean loss: ${losses.average().also { epochLosses.add(it) }}")

            }

        }

        println("seconds: ${millis / 1000.0}")

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
