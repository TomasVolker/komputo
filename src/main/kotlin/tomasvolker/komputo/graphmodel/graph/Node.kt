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
