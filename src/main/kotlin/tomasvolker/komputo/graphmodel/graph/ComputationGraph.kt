package tomasvolker.komputo.graphmodel.graph

import org.tensorflow.framework.GraphDef
import tomasvolker.komputo.graphmodel.graph.math.*
import tomasvolker.komputo.graphmodel.proto.*
import java.lang.IllegalArgumentException

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
    operator fun Operand.minus(other: Operand): Sub = subtract(this, other)
    operator fun Operand.times(other: Operand): Mul = multiply(this, other)
    operator fun Operand.div(other: Operand): Div = div(this, other)
    infix fun Operand.matMul(other: Operand): MatMul = matMul(this, other)

}