package tomasvolker.komputo.graphmodel.graph.array

import tomasvolker.komputo.graphmodel.proto.attr
import tomasvolker.komputo.graphmodel.proto.input
import tomasvolker.komputo.graphmodel.proto.nodeDef
import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import tomasvolker.komputo.graphmodel.graph.*

data class StridedSlice(
    override val name: String,
    val input: OperandRef,
    val begin: OperandRef,
    val end: OperandRef,
    val strides: OperandRef,
    override val type: DataType,
    val indexType: DataType,
    val beginMask: Long = 0,
    val endMask: Long = 0,
    val newAxisMask: Long = 0,
    val shrinkAxisMask: Long = 0
): AbstractOperandNode() {

    override fun toNodeDef(): NodeDef =
            nodeDef(operationName) {
                name = this@StridedSlice.name
                input(input)
                input(begin)
                input(end)
                input(strides)

                attr("T", type)
                attr("Index", indexType)

                attr("begin_mask", beginMask)
                attr("end_mask", endMask)
                attr("new_axis_mask", newAxisMask)
                attr("shrink_axis_mask", shrinkAxisMask)
            }

    companion object: NodeParser<StridedSlice> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "StridedSlice"

        override fun parse(nodeDef: NodeDef): StridedSlice =
                StridedSlice(
                        name = nodeDef.name,
                        input = nodeDef.getInput(0).toOperandRef(),
                        begin = nodeDef.getInput(1).toOperandRef(),
                        end = nodeDef.getInput(2).toOperandRef(),
                        strides = nodeDef.getInput(3).toOperandRef(),
                        type = nodeDef.attr("T").type,
                        indexType = nodeDef.attr("Index").type,
                        beginMask = nodeDef.attr("begin_mask").i,
                        endMask = nodeDef.attr("end_mask").i,
                        newAxisMask = nodeDef.attr("new_axis_mask").i,
                        shrinkAxisMask = nodeDef.attr("shrink_axis_mask").i
                )

    }

}

fun ScopedGraphBuilder.strideSlice(
    input: OperandRef,
    begin: OperandRef,
    end: OperandRef,
    strides: OperandRef,
    type: DataType,
    indexType: DataType,
    beginMask: Long = 0,
    endMask: Long = 0,
    newAxisMask: Long = 0,
    shrinkAxisMask: Long = 0,
    name: String? = null
): StridedSlice =
        StridedSlice(
                name = name ?: newName(Reshape.operationName),
                input = input,
                begin = begin,
                end = end,
                type = type,
                indexType = indexType,
                strides = strides,
                beginMask = beginMask,
                endMask = endMask,
                newAxisMask = newAxisMask,
                shrinkAxisMask = shrinkAxisMask
        ).also { addNode(it) }

