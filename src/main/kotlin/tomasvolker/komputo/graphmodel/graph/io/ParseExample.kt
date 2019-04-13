package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.TensorShapeProto
import tomasvolker.komputo.graphmodel.graph.*
import tomasvolker.komputo.graphmodel.proto.*

data class ParseExample(
    override val name: String,
    val input: OperandRef,
    val names: OperandRef,
    val sparseKeys: List<OperandRef> = emptyList(),
    val denseKeys: List<OperandRef> = emptyList(),
    val denseDefaults: List<OperandRef> = emptyList(),
    val sparseTypes: List<DataType> = emptyList(),
    val denseTypes: List<DataType> = emptyList(),
    val denseShapes: List<TensorShapeProto> = emptyList()
): AbstractGraphNode() {

    val sparseIndices = sparseKeys.indices.map { i ->
        NodeOutputOperand(
            node = this,
            index = i,
            type = sparseTypes[i]
        )
    }

    val sparseValues = sparseKeys.indices.map { i ->
        NodeOutputOperand(
            node = this,
            index = i,
            type = sparseTypes[i]
        )
    }

    val sparseShapes = sparseKeys.indices.map { i ->
        NodeOutputOperand(
            node = this,
            index = i,
            type = sparseTypes[i]
        )
    }

    val denseValues = denseKeys.indices.map { i ->
        NodeOutputOperand(
            node = this,
            index = i,
            type = denseTypes[i]
        )
    }

    override val outputList = sparseIndices + sparseValues + sparseShapes + denseValues

    override fun toNodeDef(): NodeDef =
        nodeDef(operationName, name) {
            input(input)

            input(names)

            sparseKeys.forEach {
                input(it)
            }

            denseKeys.forEach {
                input(it)
            }

            denseDefaults.forEach {
                input(it)
            }

            attr("Nsparse", sparseKeys.size)
            attr("Ndense", denseKeys.size)

            attr("sparse_types") {
                list = attrListValue {
                    sparseTypes.forEach {
                        addType(it)
                    }
                }
            }

            attr("Tdense") {
                list = attrListValue {
                    denseTypes.forEach {
                        addType(it)
                    }
                }
            }

            attr("dense_shapes") {
                list = attrListValue {
                    denseShapes.forEach {
                        addShape(it)
                    }
                }
            }
        }

    companion object: NodeParser<ParseExample> {

        init { GraphParser.default.register(this) }

        override val operationName: String = "ParseExample"

        override fun parse(nodeDef: NodeDef): ParseExample =
            TODO()

    }

}

fun ScopedGraphBuilder.parseExample(
    input: OperandRef,
    names: OperandRef,
    sparseKeys: List<OperandRef> = emptyList(),
    denseKeys: List<OperandRef> = emptyList(),
    denseDefaults: List<OperandRef> = emptyList(),
    sparseTypes: List<DataType> = emptyList(),
    denseTypes: List<DataType> = emptyList(),
    denseShapes: List<TensorShapeProto> = emptyList(),
    name: String? = null
): ParseExample =
    ParseExample(
        name = name ?: newName(ReadFile.operationName),
        names = names,
        sparseKeys = sparseKeys,
        denseKeys = denseKeys,
        denseDefaults = denseDefaults,
        sparseTypes = sparseTypes,
        denseTypes = denseTypes,
        denseShapes = denseShapes,
        input = input
    ).also { addNode(it) }