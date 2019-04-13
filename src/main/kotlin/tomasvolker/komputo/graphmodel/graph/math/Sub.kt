package tomasvolker.komputo.graphmodel.graph.math

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class Sub(
    override val name: String,
    val input1: OperandRef,
    val input2: OperandRef,
    override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(input1)
                input(input2)
                attr("T", type)
            }

    companion object: NodeParser<Sub> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "Sub"

        override fun parse(nodeDef: NodeDef): Sub =
            Sub(
                name = nodeDef.name,
                input1 = nodeDef.getInput(0).toOperandRef(),
                input2 = nodeDef.getInput(1).toOperandRef(),
                type = nodeDef.attr("T").type
            )

    }

}


fun ScopedGraphBuilder.subtract(
    input1: OperandRef,
    input2: OperandRef,
    type: DataType,
    name: String? = null
): Sub =
    Sub(
        name = name ?: newName(Add.operationName),
        input1 = input1,
        input2 = input2,
        type = type
    ).also { addNode(it) }


fun ScopedGraphBuilder.subtract(
    input1: Operand,
    input2: Operand,
    name: String? = null
): Sub =
    subtract(
        name = name ?: newName(Add.operationName),
        input1 = input1,
        input2 = input2,
        type = input1.type
    )
