package tomasvolker.komputo.graphmodel.graph

import com.google.protobuf.ByteString
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import java.io.InputStream
import java.nio.ByteBuffer

interface GraphNode: GraphNodeRef {

    override val name: String

    val inputList: List<OperandRef> get() = emptyList()
    val outputList: List<Operand> get() = emptyList()

    fun toNodeDef(): NodeDef

}

interface GraphNodeRef {
    val name: String
}

data class StringGraphNodeRef(
    override val name: String
): GraphNodeRef

fun String.toNodeRef() = StringGraphNodeRef(this)

abstract class AbstractGraphNode: GraphNode {

    override fun equals(other: Any?): Boolean =
            this === other ||
            other is GraphNode &&
            this.name == other.name

    override fun hashCode(): Int = name.hashCode()

    override fun toString(): String = toNodeDef().toString()

}

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

interface NodeParser<T: GraphNode> {

    val operationName: String

    fun parse(nodeDef: NodeDef): T

    fun parse(nodeDef: ByteArray): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: ByteBuffer): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: ByteString): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: InputStream): T = parse(NodeDef.parseFrom(nodeDef))

}