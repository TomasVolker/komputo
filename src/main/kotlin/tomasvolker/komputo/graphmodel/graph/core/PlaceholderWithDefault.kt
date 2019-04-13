package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class PlaceholderWithDefault(
        override val name: String,
        override val type: DataType,
        val default: OperandRef
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef("Placeholder") {
                name = this@PlaceholderWithDefault.name
                input(default)
                attr("dtype", type)
            }

    companion object: NodeParser<PlaceholderWithDefault> {

        override val operationName: String = "PlaceholderWithDefault"

        override fun parse(nodeDef: NodeDef): PlaceholderWithDefault =
                PlaceholderWithDefault(
                        name = nodeDef.name,
                        type = nodeDef.attr("dtype").type,
                        default = nodeDef.getInput(0).toOperandRef()
                )

    }

}

fun ScopedGraphBuilder.placeholderWithDefault(
    type: DataType,
    default: OperandRef,
    name: String? = null
): PlaceholderWithDefault =
        PlaceholderWithDefault(
                name = name ?: newName(PlaceholderWithDefault.operationName),
                default = default,
                type = type
        ).also { addNode(it) }
