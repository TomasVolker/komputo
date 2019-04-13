package tomasvolker.komputo.graphmodel.graph.io

import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class WriteFile(
        override val name: String,
        val filename: OperandRef,
        val contents: OperandRef
): AbstractGraphNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                input(filename)
                input(contents)
            }

    companion object: NodeParser<WriteFile> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "WriteFile"

        override fun parse(nodeDef: NodeDef): WriteFile =
            WriteFile(
                name = nodeDef.name,
                filename = nodeDef.getInput(0).toOperandRef(),
                contents = nodeDef.getInput(1).toOperandRef()
            )

    }

}

fun ScopedGraphBuilder.writeFile(
    filename: OperandRef,
    contents: OperandRef,
    name: String? = null
): WriteFile =
    WriteFile(
        name = name ?: newName(ReadFile.operationName),
        filename = filename,
        contents = contents
    ).also { addNode(it) }
