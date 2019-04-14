package tomasvolker.komputo.graphmodel.graph

import com.google.protobuf.ByteString
import org.tensorflow.Tensor
import org.tensorflow.framework.GraphDef
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.core.constant
import tomasvolker.komputo.graphmodel.graph.core.identity
import tomasvolker.komputo.graphmodel.graph.core.placeholder
import java.io.InputStream
import java.nio.ByteBuffer

fun main() {

    val graph = computationGraph {

        val input = placeholder(DT_FLOAT, name = "input")

        identity(constant(5.2f) + input, "output")
    }
/*
    graph.session {

        val result = run {
            feed("input", Tensor.create())
            fetch("output")
        }

    }
*/
}

class GraphParser {

    private val parseMap = mutableMapOf<String, NodeParser<*>>()

    fun register(nodeParser: NodeParser<*>) {
        parseMap[nodeParser.operationName] = nodeParser
    }

    fun parseGraph(graph: GraphDef): ComputationGraph =
        ComputationGraph(
            nodeSet = graph.nodeList.map {
                parseMap[it.op]?.parse(it) ?: error("default parser not implemented")
            }.toSet()
        )

    companion object {
        val default = GraphParser()
    }

}

interface NodeParser<T: GraphNode> {

    val operationName: String

    fun parse(nodeDef: NodeDef): T

    fun parse(nodeDef: ByteArray): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: ByteBuffer): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: ByteString): T = parse(NodeDef.parseFrom(nodeDef))
    fun parse(nodeDef: InputStream): T = parse(NodeDef.parseFrom(nodeDef))

}