package tomasvolker.komputo.mnist

/*
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

        val input = input(shape = I[dynamic, 28, 28])

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

        output(model)

        metric = meanSquareError(output, target)

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

                println("Mean metric: ${losses.average().also { epochLosses.add(it) }}")

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
*/