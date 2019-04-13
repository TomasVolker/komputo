package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorShapeProto
import tomasvolker.komputo.graphmodel.graph.AbstractGraphNode
import tomasvolker.komputo.graphmodel.graph.GraphParser
import tomasvolker.komputo.graphmodel.graph.NodeParser
import tomasvolker.komputo.graphmodel.graph.ScopedGraphBuilder
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.attrListValue
import tomasvolker.komputo.graphmodel.proto.nodeDef

data class TFRecordReader(
    override val name: String
): AbstractGraphNode() {

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
        }

    companion object: NodeParser<TFRecordReader> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "TFRecordReaderV2"

        override fun parse(nodeDef: NodeDef): TFRecordReader =
            TFRecordReader(
                name = nodeDef.name
            )

    }

}


fun ScopedGraphBuilder.tfRecordReader(name: String? = null): TFRecordReader =
    TFRecordReader(
        name = name ?: newName(TFRecordReader.operationName)
    ).also { addNode(it) }

