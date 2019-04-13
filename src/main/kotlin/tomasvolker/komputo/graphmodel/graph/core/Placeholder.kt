package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.graph.AbstractOperandNode
import tomasvolker.komputo.graphmodel.graph.NodeParser
import tomasvolker.komputo.graphmodel.graph.ScopedGraphBuilder
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorShapeProto
import tomasvolker.komputo.graphmodel.graph.GraphParser

data class Placeholder(
        override val name: String,
        override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                attr("dtype", type)
            }

    companion object: NodeParser<Placeholder> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "Placeholder"

        override fun parse(nodeDef: NodeDef): Placeholder =
                Placeholder(
                        name = nodeDef.name,
                        type = nodeDef.attr("dtype").type
                )

    }

}

fun ScopedGraphBuilder.placeholder(
        type: DataType,
        shape: TensorShapeProto? = null,
        name: String? = null
): Placeholder =
        Placeholder(
                name = name ?: newName(Placeholder.operationName),
                type = type
        ).also { addNode(it) }

