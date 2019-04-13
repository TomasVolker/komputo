package tomasvolker.komputo.graphmodel.graph.logging

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.graph.io.ReaderRead
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class MergeSummary(
    override val name: String,
    val inputs: List<OperandRef>
): AbstractOperandNode() {

    override val type: DataType get() = DataType.DT_STRING

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            inputs.forEach {
                input(it)
            }
        }

    companion object: NodeParser<MergeSummary> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "MergeSummary"

        override fun parse(nodeDef: NodeDef): MergeSummary =
            MergeSummary(
                name = nodeDef.name,
                inputs = List(nodeDef.inputCount) { i -> nodeDef.getInput(i).toOperandRef() }
            )

    }

}

fun ScopedGraphBuilder.mergeSummary(
    inputs: List<OperandRef>,
    name: String? = null
): MergeSummary =
    MergeSummary(
        name = name ?: newName(ReaderRead.operationName),
        inputs = inputs
    ).also { addNode(it) }

