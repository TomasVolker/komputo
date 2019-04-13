package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class Cast(
    override val name: String,
    val input: OperandRef,
    override val type: DataType,
    val truncate: Boolean? = null
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(input)
                attr("DstT", type)
                truncate?.run {
                    attr("Truncate", truncate)
                }
            }

    companion object: NodeParser<Cast> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "Cast"

        override fun parse(nodeDef: NodeDef): Cast =
                Cast(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        type = nodeDef.attr("DstT").type,
                        truncate = nodeDef.attr("Truncate", null)?.b
                )

    }

}

fun ScopedGraphBuilder.cast(
    input: OperandRef,
    type: DataType,
    truncate: Boolean? = null,
    name: String? = null
): Cast =
        Cast(
                name = name ?: newName(Cast.operationName),
                input = input,
                type = type,
                truncate = truncate
        ).also { addNode(it) }

