package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class ReaderRead(
    override val name: String,
    val reader: GraphNodeRef,
    val queue: GraphNodeRef
): AbstractGraphNode() {

    override val outputList: List<Operand> =
        listOf(
            NodeOutputOperand(
                this,
                index = 0,
                type = DataType.DT_STRING
            ),
            NodeOutputOperand(
                this,
                index = 1,
                type = DataType.DT_STRING // missing type
            )
        )

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            input(reader.name)
            input(queue.name)
        }

    companion object: NodeParser<ReaderRead> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "ReaderReadV2"

        override fun parse(nodeDef: NodeDef): ReaderRead =
            ReaderRead(
                name = nodeDef.name,
                reader = nodeDef.getInput(0).toNodeRef(),
                queue = nodeDef.getInput(1).toNodeRef()
            )

    }

}


fun ScopedGraphBuilder.readerRead(
    reader: GraphNodeRef,
    queue: GraphNodeRef,
    name: String? = null
): ReaderRead =
    ReaderRead(
        name = name ?: newName(ReaderRead.operationName),
        reader = reader,
        queue = queue
    ).also { addNode(it) }

