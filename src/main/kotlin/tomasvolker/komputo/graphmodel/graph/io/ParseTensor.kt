package tomasvolker.komputo.graphmodel.graph.io

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class ParseTensor(
    override val name: String,
    val input: OperandRef,
    override val type: DataType
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@ParseTensor.name
                input(input)
                attr("out_type", type)
            }

    companion object: NodeParser<ParseTensor> {

        override val operationName: String = "ParseTensor"

        override fun parse(nodeDef: NodeDef): ParseTensor =
                ParseTensor(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        type = nodeDef.attr("out_type").type
                )

    }

}

fun ScopedGraphBuilder.parseTensor(
    input: OperandRef,
    type: DataType,
    name: String? = null
): ParseTensor =
        ParseTensor(
                name = name ?: newName(ReadFile.operationName),
                type = type,
                input = input
        ).also { addNode(it) }
