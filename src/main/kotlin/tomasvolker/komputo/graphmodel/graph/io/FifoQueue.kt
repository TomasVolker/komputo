package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorShapeProto
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.attrListValue
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class FifoQueue(
    override val name: String,
    val componentTypes: List<DataType>,
    val componentShapes: List<TensorShapeProto>? = null,
    val capacity: Long? = null,
    val container: String? = null,
    val sharedName: String? = null
): AbstractGraphNode() {

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName) {
            name = this@FifoQueue.name
            attr("component_types") {
                list = attrListValue {
                    componentTypes.forEach { addType(it) }
                }
            }

            componentShapes?.let {
                attr("shapes") {
                    list = attrListValue {
                        componentShapes.forEach { addShape(it) }
                    }
                }
            }

            capacity?.let { attr("capacity", it) }
            container?.let { attr("container", it) }
            sharedName?.let { attr("shared_name", it) }
        }

    companion object: NodeParser<FifoQueue> {

        override val operationName: String = "FIFOQueueV2"

        override fun parse(nodeDef: NodeDef): FifoQueue =
            FifoQueue(
                name = nodeDef.name,
                componentTypes = nodeDef.attr("out_type").list.let {
                    List(it.typeCount) { i -> it.getType(i) }
                },
                componentShapes = nodeDef.attr("shapes", null)?.list?.let {
                    List(it.shapeCount) { i -> it.getShape(i) }
                },
                capacity = nodeDef.attr("capacity", null)?.i,
                container = nodeDef.attr("container", null)?.s?.toString(Charsets.UTF_8),
                sharedName = nodeDef.attr("shared_name", null)?.s?.toString(Charsets.UTF_8)
            )

    }

}

fun ScopedGraphBuilder.fifoQueue(
    componentTypes: List<DataType>,
    componentShapes: List<TensorShapeProto>? = null,
    capacity: Long? = null,
    container: String? = null,
    sharedName: String? = null,
    name: String? = null
): FifoQueue =
    FifoQueue(
        name = name ?: newName(ReadFile.operationName),
        componentTypes = componentTypes,
        componentShapes = componentShapes,
        capacity = capacity,
        container = container,
        sharedName = sharedName
    ).also { addNode(it) }


