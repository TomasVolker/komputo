package tomasvolker.komputo.graphmodel.graph.math

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class MatMul(
    override val name: String,
    val input1: OperandRef,
    val input2: OperandRef,
    override val type: DataType,
    val transpose1: Boolean = false,
    val transpose2: Boolean = false
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(input1)
                input(input2)
                attr("T", type)
                attr("transpose_a", transpose1)
                attr("transpose_b", transpose2)
            }

    companion object: NodeParser<MatMul> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "MatMul"

        override fun parse(nodeDef: NodeDef): MatMul =
            MatMul(
                name = nodeDef.name,
                input1 = nodeDef.getInput(0).toOperandRef(),
                input2 = nodeDef.getInput(1).toOperandRef(),
                type = nodeDef.attr("T").type,
                transpose1 = nodeDef.attr("transpose_a", null)?.b ?: false,
                transpose2 = nodeDef.attr("transpose_b", null)?.b ?: false
            )

    }

}


fun ScopedGraphBuilder.matMul(
    input1: OperandRef,
    input2: OperandRef,
    type: DataType,
    transpose1: Boolean = false,
    transpose2: Boolean = false,
    name: String? = null
): MatMul =
    MatMul(
        name = name ?: newName(Add.operationName),
        input1 = input1,
        input2 = input2,
        type = type,
        transpose1 = transpose1,
        transpose2 = transpose2
    ).also { addNode(it) }


fun ScopedGraphBuilder.matMul(
    input1: Operand,
    input2: Operand,
    transpose1: Boolean = false,
    transpose2: Boolean = false,
    name: String? = null
): MatMul =
    matMul(
        name = name,
        input1 = input1,
        input2 = input2,
        type = input1.type,
        transpose1 = transpose1,
        transpose2 = transpose2
    )
