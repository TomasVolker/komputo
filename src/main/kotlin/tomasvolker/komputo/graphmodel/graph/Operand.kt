package tomasvolker.komputo.graphmodel.graph

import org.tensorflow.framework.DataType

interface Operand: OperandRef {
    override val name: String
    val type: DataType
}

interface OperandRef {
    val name: String
}

data class NodeOutputOperand<T: GraphNode>(
    val node: T,
    val index: Int,
    override val type: DataType
): Operand {

    override val name: String
        get() = "${node.name}:$index"

}

data class StringOperandRef(
    override val name: String
): OperandRef

fun String.toOperandRef() = StringOperandRef(this)

interface OperandNode: GraphNode, Operand {

    override val outputList: List<Operand>
        get() = listOf(this)

}

abstract class AbstractOperandNode: AbstractGraphNode(), OperandNode