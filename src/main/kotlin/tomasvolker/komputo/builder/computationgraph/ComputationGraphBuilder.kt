package tomasvolker.komputo.builder.computationgraph

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.op.Scope
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.TFOperation
import tomasvolker.komputo.dsl.get
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import java.lang.IllegalArgumentException
import java.util.regex.Pattern

interface ComputationGraph {

    val tfGraph: Graph

    val initialization: TFOperation?
    val placeholderList: List<TFOperand>
    val variableList: List<TFOperand>

}

private val NAME_REGEX = "[A-Za-z0-9.][A-Za-z0-9_.\\-]*".toRegex()

class ComputationGraphScope(
    val name: String
) {

    val nameMap = mutableMapOf<String, Int>()

    fun newName(name: String): String {
        if(name.matches(NAME_REGEX)) {
            val index = nameMap[name] ?: 0
            nameMap[name] = index + 1
            return name + if(index > 0) "_$index" else ""
        } else {
            throw IllegalArgumentException("The name provided is not valid")
        }
    }

}

interface ComputationGraphBuilder {

    val tfGraph: Graph

    var defaultDataType: DataType
    var defaultFloatDataType: DataType
    var defaultIntegerDataType: DataType

}

inline fun ComputationGraphBuilder.operation(
    operation: String,
    init: TFOperationBuilder.()->Unit
): TFOperation =
    TFOperationBuilder(this, operation).apply(init).build()



inline fun ComputationGraphBuilder.operand(
    operation: String,
    init: TFOperationBuilder.()->Unit
): TFOperand = operation(operation, init)[0]



fun ComputationGraphBuilder.placeholder(
    shape: IntArray1D? = null,
    dataType: DataType = defaultDataType,
    name: String? = null
) = operand("Placeholder") {

    nodeName = name

    dtype = dataType

    shape?.let {
        this.shape = shape
    }

}


fun ComputationGraphBuilder.variable(
    shape: IntArray1D? = null,
    dataType: DataType = defaultDataType,
    name: String? = null
) = operand("VariableV2") {

    nodeName = name

    dtype = dataType

    shape?.let {
        this.shape = shape
    }

}
