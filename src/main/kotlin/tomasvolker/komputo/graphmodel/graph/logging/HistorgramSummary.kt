package tomasvolker.komputo.graphmodel.graph.logging

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.graph.io.ReaderRead
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class HistogramSummary(
    override val name: String,
    val tag: OperandRef,
    val values: OperandRef
): AbstractOperandNode() {

    override val type: DataType get() = DataType.DT_STRING

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            input(tag)
            input(values)
        }

    companion object: NodeParser<HistogramSummary> {

        override val operationName: String = "HistogramSummary"

        override fun parse(nodeDef: NodeDef): HistogramSummary =
            HistogramSummary(
                name = nodeDef.name,
                tag = nodeDef.getInput(0).toOperandRef(),
                values = nodeDef.getInput(1).toOperandRef()
            )

    }

}

fun ScopedGraphBuilder.histogramSummary(
    tags: OperandRef,
    values: OperandRef,
    name: String? = null
): HistogramSummary =
    HistogramSummary(
        name = name ?: newName(ReaderRead.operationName),
        tag = tags,
        values = values
    ).also { addNode(it) }

