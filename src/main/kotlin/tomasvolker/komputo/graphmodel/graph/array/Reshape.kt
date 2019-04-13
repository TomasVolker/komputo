package tomasvolker.komputo.graphmodel.graph.array

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class Reshape(
    override val name: String,
    val input: OperandRef,
    val shape: OperandRef,
    override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@Reshape.name
                input(input)
                input(shape)
                attr("T", type)
            }

    companion object: NodeParser<Reshape> {

        override val operationName: String = "Reshape"

        override fun parse(nodeDef: NodeDef): Reshape =
                Reshape(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        shape = nodeDef.getInput(1).toOperandRef(),
                        type = nodeDef.attr("T").type
                )

    }

}

fun ScopedGraphBuilder.reshape(
    input: OperandRef,
    shape: OperandRef,
    type: DataType,
    name: String? = null
): Reshape =
        Reshape(
                name = name ?: newName(Reshape.operationName),
                input = input,
                shape = shape,
                type = type
        ).also { addNode(it) }

