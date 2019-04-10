package tomasvolker.komputo.builder.computationgraph

import org.tensorflow.DataType
import org.tensorflow.Graph
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.TFOperation
import tomasvolker.komputo.dsl.get
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D

interface ComputationGraph {

    val tfGraph: Graph

    val initialization: TFOperation?
    val placeholderList: List<TFOperand>
    val variableList: List<TFOperand>

    class Scope(
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

    fun operation(name: String): TFOperation? = tfGraph.operation(name)
    val operations: Set<TFOperation> get() = tfGraph.operations().asSequence().toSet()

}

private val NAME_REGEX = "[A-Za-z0-9.][A-Za-z0-9_.\\-]*".toRegex()



interface ComputationGraphBuilder {

    val tfGraph: Graph

    var defaultDataType: DataType
    var defaultFloatDataType: DataType
    var defaultIntegerDataType: DataType

    var scope: ComputationGraph.Scope

    fun build(): ComputationGraph

}

class ComputationGraphBuilderImpl(
    override val tfGraph: Graph = Graph()
) : ComputationGraphBuilder {

    override var defaultDataType = DataType.FLOAT
    override var defaultFloatDataType = DataType.FLOAT
    override var defaultIntegerDataType = DataType.FLOAT

    override var scope = ComputationGraph.Scope("")

    override fun build(): ComputationGraph = TODO()

}

fun computationGraph(block: ComputationGraphBuilder.()->Unit): ComputationGraph =
        ComputationGraphBuilderImpl().apply(block).build()

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
