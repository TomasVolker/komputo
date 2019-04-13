package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.NodeOutput
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.attrListValue
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class QueueDequeue(
    override val name: String,
    val queue: GraphNodeRef,
    val componentTypes: List<DataType>
): AbstractGraphNode() {

    override val outputList: List<Operand> =
        componentTypes.mapIndexed { i, type ->
            NodeOutputOperand(
                node = this,
                index = i,
                type = type
            )
        }

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName) {
            name = this@QueueDequeue.name
            input(queue.name)
            attr("component_types") {
                list = attrListValue {
                    componentTypes.forEach { addType(it) }
                }
            }
        }

    companion object: NodeParser<QueueDequeue> {

        override val operationName: String = "QueueDequeueV2"

        override fun parse(nodeDef: NodeDef): QueueDequeue =
            QueueDequeue(
                name = nodeDef.name,
                queue = nodeDef.getInput(0).toNodeRef(),
                componentTypes = nodeDef.attr("Tcomponents").list.let {
                    List(it.typeCount) { i -> it.getType(i) }
                }
            )

    }

}

fun ScopedGraphBuilder.queueDequeue(
    queue: GraphNodeRef,
    componentTypes: List<DataType>,
    name: String? = null
): QueueDequeue =
    QueueDequeue(
        name = name ?: newName(QueueEnqueue.operationName),
        queue = queue,
        componentTypes = componentTypes
    ).also { addNode(it) }

/*
fun ScopedGraphBuilder.dequeue(
        queue: NodeDef,
        dataTypeList: List<DataType>,
        name: String? = null
): NodeDef =
        node("QueueDequeueV2", name) {
            input(queue)
            attr("component_types") {
                list = attrListValue {
                    dataTypeList.forEach { addType(it) }
                }
            }
        }

*/