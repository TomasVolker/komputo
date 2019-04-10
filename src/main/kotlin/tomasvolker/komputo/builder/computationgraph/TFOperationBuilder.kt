package tomasvolker.komputo.builder.computationgraph

import org.tensorflow.*
import org.tensorflow.DataType
import org.tensorflow.framework.*
import org.tensorflow.op.core.Gradients
import tomasvolker.komputo.TFOperation
import tomasvolker.komputo.TFOutput
import tomasvolker.komputo.dsl.buildOp
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.generic.ArrayND
import java.lang.IllegalArgumentException
import java.nio.IntBuffer

class TFOperationBuilder(
    val builder: ComputationGraphBuilder,
    val operation: String
) {

    val scope: ComputationGraph.Scope get() = builder.scope

    val attributes = mutableMapOf<String, Any?>()
    val controlOperations = mutableListOf<TFOperation>()
    val inputs = mutableListOf<TFOutput>()

    var nodeName: String? = null

    var device: String? = null

    private fun String.setAttr(value: Any?) {
        attributes[this] = value
    }

    infix fun String.setTo(value: DataType?) = setAttr(value)
    infix fun String.setTo(value: String?) = setAttr(value)
    infix fun String.setTo(value: ArrayND<*>?) = setAttr(value)

    fun build(): TFOperation = builder.tfGraph.buildOp(operation, scope.newName(nodeName ?: operation)) {
        attributes.forEach { name, value -> setAttr(name, value) }
        controlOperations.forEach { addControlInput(it) }
        inputs.forEach { addInput(it) }
        device?.let { setDevice(it) }
    }

}

fun OperationBuilder.setAttr(name: String, value: Any?): OperationBuilder = when(value) {
    is String  -> setAttr(name, value)
    is ByteArray  -> setAttr(name, value)
    is Long  -> setAttr(name, value)
    is LongArray  -> setAttr(name, value)
    is Float  -> setAttr(name, value)
    is FloatArray  -> setAttr(name, value)
    is Boolean  -> setAttr(name, value)
    is BooleanArray  -> setAttr(name, value)
    is DataType  -> setAttr(name, value)
    is Tensor<*> -> setAttr(name, value)
    is Shape -> setAttr(name, value)
    is IntArray1D -> value.asTensor { setAttr(name, it) }
    else -> throw IllegalArgumentException("invalid value type")
}

fun <T> IntArray1D.asTensor(block: (Tensor<Int>)->T): T =
    Tensor.create(
        LongArray(rank) { i -> shape(i).toLong() },
        IntBuffer.wrap(toIntArray())
    ).use(block)


var TFOperationBuilder.dtype: DataType?
    get() = attributes["dtype"] as DataType?
    set(value) { "dtype" setTo value }

var TFOperationBuilder.shape: IntArray1D?
    get() = attributes["shape"] as IntArray1D?
    set(value) { "shape" setTo value }
