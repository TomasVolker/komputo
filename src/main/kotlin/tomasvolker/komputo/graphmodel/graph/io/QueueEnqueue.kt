package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.attrListValue
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class QueueEnqueue(
    override val name: String,
    val queue: GraphNodeRef,
    val values: List<OperandRef>,
    val componentTypes: List<DataType>
): AbstractGraphNode() {

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            input(queue.name)
            values.forEach {
                input(it)
            }
            attr("Tcomponents") {
                list = attrListValue {
                    componentTypes.forEach { addType(it) }
                }
            }
        }

    companion object: NodeParser<QueueEnqueue> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "QueueEnqueueV2"

        override fun parse(nodeDef: NodeDef): QueueEnqueue =
            QueueEnqueue(
                name = nodeDef.name,
                queue = nodeDef.getInput(0).toNodeRef(),
                values = List(nodeDef.inputCount-1) { i -> nodeDef.getInput(i+1).toOperandRef() },
                componentTypes = nodeDef.attr("Tcomponents").list.let {
                    List(it.typeCount) { i -> it.getType(i) }
                }
            )

    }

}

fun ScopedGraphBuilder.queueEnqueue(
    queue: GraphNodeRef,
    values: List<OperandRef>,
    componentTypes: List<DataType>,
    name: String? = null
): QueueEnqueue =
    QueueEnqueue(
        name = name ?: newName(QueueEnqueue.operationName),
        queue = queue,
        values = values,
        componentTypes = componentTypes
    ).also { addNode(it) }

fun ScopedGraphBuilder.queueEnqueue(
    queue: Queue,
    values: List<OperandRef>,
    name: String? = null
): QueueEnqueue =
    QueueEnqueue(
        name = name ?: newName(QueueEnqueue.operationName),
        queue = queue,
        values = values,
        componentTypes = queue.componentTypes
    ).also { addNode(it) }