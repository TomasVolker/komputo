package tomasvolker.komputo.graphmodel.graph.array

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class Slice(
    override val name: String,
    val input: OperandRef,
    val begin: OperandRef,
    val size: OperandRef,
    override val type: DataType,
    val indexType: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@Slice.name
                input(input)
                input(begin)
                input(size)

                attr("T", type)
                attr("Index", indexType)
            }

    companion object: NodeParser<Slice> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "Slice"

        override fun parse(nodeDef: NodeDef): Slice =
                Slice(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        begin = nodeDef.getInput(1).toOperandRef(),
                        size = nodeDef.getInput(2).toOperandRef(),
                        type = nodeDef.attr("T").type,
                        indexType = nodeDef.attr("Index").type
                )

    }

}

fun ScopedGraphBuilder.slice(
    input: OperandRef,
    begin: OperandRef,
    size: OperandRef,
    type: DataType,
    indexType: DataType,
    name: String? = null
): Slice =
        Slice(
                name = name ?: newName(ExpandDims.operationName),
                input = input,
                begin = begin,
                size = size,
                type = type,
                indexType = indexType
        ).also { addNode(it) }
