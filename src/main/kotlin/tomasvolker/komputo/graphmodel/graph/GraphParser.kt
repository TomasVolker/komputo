package tomasvolker.komputo.graphmodel.graph

import com.google.protobuf.ByteString
import org.tensorflow.framework.GraphDef
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.core.constant
import java.io.InputStream
import java.nio.ByteBuffer

fun main() {

    val graphdef = computationGraph {
        constant(5) + constant(6)
    }.toGraphDef()

    GraphParser.default.parseGraph(graphdef).also { println(it) }

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