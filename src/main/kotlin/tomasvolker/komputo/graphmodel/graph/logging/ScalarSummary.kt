package tomasvolker.komputo.graphmodel.graph.logging

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.graph.io.ReaderRead
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class ScalarSummary(
    override val name: String,
    val tags: OperandRef,
    val values: OperandRef
): AbstractOperandNode() {

    override val type: DataType get() = DataType.DT_STRING

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            input(tags)
            input(values)
        }

    companion object: NodeParser<ScalarSummary> {

        override val operationName: String = "ScalarSummary"

        override fun parse(nodeDef: NodeDef): ScalarSummary =
            ScalarSummary(
                name = nodeDef.name,
                tags = nodeDef.getInput(0).toOperandRef(),
                values = nodeDef.getInput(1).toOperandRef()
            )

    }

}

fun ScopedGraphBuilder.scalarSummary(
    tags: OperandRef,
    values: OperandRef,
    name: String? = null
): ScalarSummary =
    ScalarSummary(
        name = name ?: newName(ReaderRead.operationName),
        tags = tags,
        values = values
    ).also { addNode(it) }

