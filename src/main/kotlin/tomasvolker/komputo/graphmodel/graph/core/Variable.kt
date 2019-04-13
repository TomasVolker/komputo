package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.graph.AbstractOperandNode
import tomasvolker.komputo.graphmodel.graph.NodeParser
import tomasvolker.komputo.graphmodel.graph.ScopedGraphBuilder
import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorShapeProto

class Variable(
        override val name: String,
        override val type: DataType,
        val shape: TensorShapeProto,
        val container: String? = null,
        val sharedName: String? = null
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName, name) {
                attr("dtype", type)
                attr("shape", shape)
                container?.let { attr("container", it) }
                sharedName?.let { attr("shared_name", it) }
            }

    companion object: NodeParser<Variable> {

        override val operationName: String = "Variable"

        override fun parse(nodeDef: NodeDef): Variable =
                Variable(
                        name = nodeDef.name,
                        type = nodeDef.attr("dtype").type,
                        shape = nodeDef.attr("shape").shape,
                        container = nodeDef.attr("container", null)?.s?.toString(Charsets.UTF_8),
                        sharedName = nodeDef.attr("shared_name", null)?.s?.toString(Charsets.UTF_8)
                )

    }

}


fun ScopedGraphBuilder.variable(
        type: DataType,
        shape: TensorShapeProto,
        container: String? = null,
        sharedName: String? = null,
        name: String? = null
): Variable =
        Variable(
                name = name ?: newName(Variable.operationName),
                type = type,
                shape = shape,
                container = container,
                sharedName = sharedName
        ).also { addNode(it) }
