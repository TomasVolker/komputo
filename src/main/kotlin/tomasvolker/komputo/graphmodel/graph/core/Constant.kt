package tomasvolker.komputo.graphmodel.graph.core

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.nodeDef
import tomasvolker.komputo.graphmodel.proto.tensorProtoScalar
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorProto
import tomasvolker.komputo.graphmodel.graph.AbstractOperandNode
import tomasvolker.komputo.graphmodel.graph.NodeParser
import tomasvolker.komputo.graphmodel.graph.ScopedGraphBuilder

data class Constant(
        override val name: String,
        val value: TensorProto // change to numeriko value
): AbstractOperandNode() {

    override val type: DataType get() = value.dtype

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@Constant.name
                attr("dtype", value.dtype)
                attr("value", value)
            }

    companion object: NodeParser<Constant> {

        override val operationName: String = "Const"

        override fun parse(nodeDef: NodeDef): Constant =
                Constant(
                    name = nodeDef.name,
                    value = nodeDef.attr("value").tensor
                )

    }

}

fun ScopedGraphBuilder.constant(
        value: TensorProto,
        name: String? = null
): Constant =
        Constant(
            name = name ?: newName(Constant.operationName),
            value = value
        ).also { addNode(it) }

fun ScopedGraphBuilder.constant(
        value: Float,
        type: DataType = DataType.DT_FLOAT,
        name: String? = null
): Constant = constant(tensorProtoScalar(value, type), name)

fun ScopedGraphBuilder.constant(
        value: Double,
        type: DataType = DataType.DT_DOUBLE,
        name: String? = null
): Constant = constant(tensorProtoScalar(value, type), name)

fun ScopedGraphBuilder.constant(
        value: Int,
        type: DataType = DataType.DT_INT32,
        name: String? = null
): Constant = constant(tensorProtoScalar(value, type), name)

fun ScopedGraphBuilder.constant(
        value: Long,
        type: DataType = DataType.DT_INT64,
        name: String? = null
): Constant = constant(tensorProtoScalar(value, type), name)

fun ScopedGraphBuilder.constant(
        value: String,
        name: String? = null
): Constant = constant(tensorProtoScalar(value), name)



