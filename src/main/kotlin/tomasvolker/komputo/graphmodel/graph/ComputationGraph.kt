package tomasvolker.komputo.graphmodel.graph

import org.tensorflow.framework.DataType
import tomasvolker.komputo.graphmodel.graph.core.Add
import tomasvolker.komputo.graphmodel.graph.core.add
import tomasvolker.komputo.graphmodel.graph.core.constant
import tomasvolker.komputo.graphmodel.runtime.readBytes
import tomasvolker.komputo.graphmodel.utils.asFloatBuffer
import org.tensorflow.framework.GraphDef
import tomasvolker.komputo.graphmodel.graph.array.expandDims
import tomasvolker.komputo.graphmodel.graph.array.reshape
import tomasvolker.komputo.graphmodel.graph.io.*
import tomasvolker.komputo.graphmodel.proto.*
import tomasvolker.komputo.graphmodel.record.tfRecordWriter
import tomasvolker.komputo.graphmodel.utils.toByteString
import java.io.File
import java.lang.IllegalArgumentException

fun main() {

    File("data/small.tfrecord").tfRecordWriter().use {
        it.write(
            exampleProto {
                features {
                    feature("value", 5.0)
                }
            }.toByteArray()
        )
    }


    val graph = computationGraph {

        val filesQueue = fifoQueue(listOf(DataType.DT_STRING), name = "filesQueue")

        val filename = constant("./data/small.tfrecord")

        val enqueue = queueEnqueue(filesQueue, listOf(filename), listOf(DataType.DT_STRING), "enqueueFile")

        val dequeue = queueDequeue(filesQueue, listOf(DataType.DT_STRING), "dequeueFile")

        val recordReader = tfRecordReader("reader")

        val read = readerRead(recordReader, filesQueue, "read")

        val expand = expandDims(read.outputList[1], constant(0), DataType.DT_STRING, "expand")

        val parsed = parseExample(
            input = expand,
            names = constant(
                tensorProto {
                    dtype = DataType.DT_STRING
                    shape(1)
                    addStringVal("value".toByteString())
                }
            ),
            denseKeys = listOf(
                constant("value")
            ),
            denseDefaults = listOf(
                constant(
                    tensorProto {
                        dtype = DataType.DT_FLOAT
                        shape(0)
                    }
                )
            ),
            denseTypes = listOf(DataType.DT_FLOAT),
            denseShapes = listOf(tensorShapeProto { }),
            name = "parseExample"
        )
/*
        val reshape = reshape(
            input = parsed.denseValues[0],
            shape = constant(
                    tensorProto {
                        dtype = DataType.DT_INT32
                        shape(0)
                    }
            ),
            type = DataType.DT_STRING
        )

        val tensor = parseTensor(reshape, DataType.DT_FLOAT, "parseTensor")
*/
    }.also { println(it) }

    graph.session {

        run {
            addTarget("enqueueFile")
        }

        val result = run {
            fetch("parseExample")
        }

        println(result[0])

        result[0].readBytes().asFloatBuffer().also {
            println(it[0])
        }

    }

}

fun computationGraph(init: ScopedGraphBuilder.() -> Unit): ComputationGraph =
        GraphBuilder().apply { scoped().apply(init) }.build()

data class ComputationGraph(
        val nodeSet: Set<GraphNode>
) {

    fun toGraphDef(): GraphDef =
            graphDef {
                nodeSet.forEach {
                    addNode(it.toNodeDef())
                }
            }

    override fun toString(): String =
            buildString {
                appendln("ComputationGraph {")
                nodeSet.forEach { appendln(it) }
                appendln("}")
            }

}

class GraphBuilder{

    private val nodeMap = mutableMapOf<String, GraphNode>()

    val nameSet: Set<String> get() = nodeMap.keys

    fun newName(name: String): String {

        if (name !in nameSet) return name

        //TODO: reduce to order 1
        var index = 1
        var newName = "${name}_$index"
        while(newName in nameSet) {
            index++
            newName = "${name}_$index"
        }

        return newName
    }

    fun addNode(node: GraphNode) {

        if (node.name in nameSet)
            error("Duplicate name ${node.name} in graph")

        nodeMap[node.name] = node
    }

    fun scoped(name: String = "") = ScopedGraphBuilder(this, name)

    fun dereference(reference: OperandRef): Operand {
        val components = reference.name.split(":")

        return when(components.size) {
            1 -> nodeMap[components[0]]?.outputList?.get(0) // should it check if it is Operand?
            2 -> nodeMap[components[0]]?.outputList?.get(components[1].toInt())
            else -> null
        } ?: throw IllegalArgumentException("Unresolved reference $reference")
    }

    fun build(): ComputationGraph =
            ComputationGraph(nodeMap.values.toSet())

}

class ScopedGraphBuilder(
    val builder: GraphBuilder,
    val scopeName: String
) {

    fun dereference(reference: OperandRef): Operand = builder.dereference(reference)

    fun OperandRef.deref(): Operand = dereference(this)
    fun String.deref(): Operand = dereference(this.toOperandRef())

    fun newName(name: String): String = builder.newName(scopeName + name)

    fun addNode(node: GraphNode) { builder.addNode(node) }

    fun scope(name: String, block: ScopedGraphBuilder.()->Unit) {
        ScopedGraphBuilder(builder,"$scopeName$name/").block()
    }

    operator fun Operand.plus(other: Operand): Add = add(this, other)

}