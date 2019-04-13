package tomasvolker.komputo.graphmodel.graph.array

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.graph.core.constant

data class ExpandDims(
    override val name: String,
    val input: OperandRef,
    val axis: OperandRef,
    override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@ExpandDims.name
                input(input)
                input(axis)
                attr("T", type)
            }

    companion object: NodeParser<ExpandDims> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "ExpandDims"

        override fun parse(nodeDef: NodeDef): ExpandDims =
                ExpandDims(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        axis = nodeDef.getInput(1).toOperandRef(),
                        type = nodeDef.attr("T").type
                )

    }

}

fun ScopedGraphBuilder.expandDims(
    input: OperandRef,
    axis: OperandRef,
    type: DataType,
    name: String? = null
): ExpandDims =
        ExpandDims(
                name = name ?: newName(ExpandDims.operationName),
                input = input,
                axis = axis,
                type = type
        ).also { addNode(it) }

fun ScopedGraphBuilder.expandDims(
    input: Operand,
    axis: OperandRef,
    name: String? = null
): ExpandDims = expandDims(input, axis, input.type, name)

fun ScopedGraphBuilder.expandDims(
    input: Operand,
    axis: Int,
    name: String? = null
): ExpandDims = expandDims(input, constant(axis), name)