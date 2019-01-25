package tomasvolker.komputo.builder.computationgraph

import org.tensorflow.DataType
import tomasvolker.komputo.TFOperation
import tomasvolker.komputo.TFOutput
import tomasvolker.komputo.dsl.buildOp
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.generic.ArrayND

class TFOperationBuilder(
    val builder: ComputationGraphBuilder,
    val operation: String
) {

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

    fun build(): TFOperation = builder.tfGraph.buildOp(operation, nodeName) {

    }

}

var TFOperationBuilder.dtype: DataType?
    get() = attributes["dtype"] as DataType?
    set(value) { "dtype" setTo value }

var TFOperationBuilder.shape: IntArray1D?
    get() = attributes["shape"] as IntArray1D?
    set(value) { "shape" setTo value }
