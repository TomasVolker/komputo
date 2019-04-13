package tomasvolker.komputo.graphmodel.graph.io

import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class ReadFile(
        override val name: String,
        val input: OperandRef
): AbstractOperandNode() {

    override val type: DataType get() = DataType.DT_STRING

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(input)
            }

    companion object: NodeParser<ReadFile> {

        override val operationName: String = "ReadFile"

        override fun parse(nodeDef: NodeDef): ReadFile =
                ReadFile(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef()
                )

    }

}

fun ScopedGraphBuilder.readFile(
    input: OperandRef,
    name: String? = null
): ReadFile =
        ReadFile(
                name = name ?: newName(ReadFile.operationName),
                input = input
        ).also { addNode(it) }
