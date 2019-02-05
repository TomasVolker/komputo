package tomasvolker.komputo.mnist

import tomasvolker.komputo.dataset.LabeledDataset
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.interfaces.factory.doubleArray1D
import tomasvolker.numeriko.core.primitives.indicator

fun highestPredictionsString(probabilities: DoubleArray1D) =
    probabilities.toList()
        .withIndex()
        .sortedByDescending { it.value }
        .take(3)
        .joinToString(prefix = "{", postfix = "}") { "%d:%.2f%%".format(it.index, it.value*100) }


fun Int.toOneHot(size: Int): DoubleArray1D =
    doubleArray1D(size) { i -> (i == this).indicator() }



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