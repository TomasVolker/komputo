package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class Identity(
    override val name: String,
    val input: OperandRef,
    override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(input)
                attr("T", type)
            }

    companion object: NodeParser<Identity> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "Identity"

        override fun parse(nodeDef: NodeDef): Identity =
            Identity(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        type = nodeDef.attr("T").type
                )

    }

}

fun ScopedGraphBuilder.identity(
    input: OperandRef,
    type: DataType,
    name: String? = null
): Identity =
    Identity(
            name = name ?: newName(Identity.operationName),
            input = input,
            type = type
    ).also { addNode(it) }

fun ScopedGraphBuilder.identity(
    input: Operand,
    name: String? = null
): Identity = identity(input, input.type, name)
