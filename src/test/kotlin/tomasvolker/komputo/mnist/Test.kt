package tomasvolker.komputo.mnist

import org.tensorflow.Graph
import org.tensorflow.Session
import tomasvolker.komputo.dsl.session
import tomasvolker.komputo.dsl.toDoubleNDArray
import tomasvolker.komputo.dsl.toTensor
import tomasvolker.komputo.functions.softmax
import tomasvolker.komputo.loadGraphDef
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array2d.generic.get
import tomasvolker.numeriko.core.interfaces.factory.array0D
import java.io.File

fun main() {

    val (train, test) = loadMnistDataset()

    val image = test.random().data

    val graph = loadGraphDef("simplenet.pb")

    graph.operations().asSequence().forEach {
        println(it)
    }

    session(graph) {

        execute {
            feed("save/file_name" to array0D("test_save.pb"))
            target("save/Group")
        }

        val prediction = execute {
            feed("Placeholder" to image.higherRank())
            fetch("dense_layer_1/Add")
        }.first().as2D()[0, All]

        val probabilities = softmax(prediction)
            .withIndex()
            .sortedByDescending { it.value }
            .map { "%d: %.2f%%".format(it.index, it.value * 100) }

        print(Mnist.renderToString(image))
        println("prediction: ${probabilities}")


    }



}
