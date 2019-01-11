package tomasvolker.komputo.dsl

import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D

open class Activation

object Sigmoid: Activation()
object Relu: Activation()
object Identity: Activation()

class SequentialModel

interface Layer

class DenseLayer: Layer {

    class Builder(
        var size: Int = 1,
        var activation: Activation = Identity
    )

}

class Conv2DLayer: Layer {

    class Builder(
        var filterCount: Int = 1,
        var stride: Int,
        var filterSize: Int,
        var activation: Activation = Identity
    )

}

class SequentialModelBuilder() {

    fun dense(
        size: Int = 1,
        init: DenseLayer.Builder.()->Unit = {}
    ): DenseLayer = TODO()

    fun flatten(): DenseLayer = TODO()

    fun reshape(shape: IntArray1D): DenseLayer = TODO()
    fun reshape(vararg shape: Int): DenseLayer = TODO()

    fun conv2D(
        init: Conv2DLayer.Builder.()->Unit = {}
    ): Conv2DLayer = TODO()

}

fun sequentialModel(vararg inputShape: Int, init: SequentialModelBuilder.()->Unit): SequentialModel = TODO()



fun main() {

    val model = sequentialModel(100) {


        dense {
            size = 100
            activation = Sigmoid
        }


        dense(100)

        flatten()

        reshape(10, 10)

        conv2D {
            activation = Relu
        }

    }


}