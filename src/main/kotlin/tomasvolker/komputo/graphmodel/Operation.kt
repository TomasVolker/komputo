package tomasvolker.komputo.graphmodel

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.core.Add as TensorflowAdd
import org.tensorflow.op.core.Placeholder as TensorflowPlaceholder
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.intArray1DOf

class Input(
    override val scope: Scope,
    override val name: String,
    override val shape: IntArray1D,
    override val type: DataType
): ComputationGraph.Expression

class Variable(
    override val scope: Scope,
    override val name: String,
    override val shape: IntArray1D,
    override val type: DataType
): ComputationGraph.Expression

class Assign(
    override val scope: Scope,
    override val name: String,
    val variable: Variable,
    val value: ComputationGraph.Expression
): ComputationGraph.Node

class RealConstant(
    override val scope: Scope,
    override val name: String,
    val value: DoubleArrayND,
    override val type: DataType
): ComputationGraph.Expression {

    override val shape: IntArray1D
        get() = value.shape

}

class Add(
    override val scope: Scope,
    override val name: String,
    val input1: ComputationGraph.Expression,
    val input2: ComputationGraph.Expression
): ComputationGraph.Expression {

    init {
        require(input1.shape == input2.shape)
        require(input1.type == input2.type)
    }

    override val shape: IntArray1D
        get() = input1.shape

    override val type: DataType
        get() = input1.type

}

class Gradient(
    val value: List<ComputationGraph.Expression>,
    val wrt: List<ComputationGraph.Expression>
) {



}




class ComputationGraph(
    val nodeList: List<Node>
) {

    interface Node {

        val name: String

        val scope: Scope

    }

    interface Expression: Node {
        val shape: IntArray1D
        val type: DataType
    }

}
/*
class ComputationGraphBuilder(
    val scope: Scope = RootScope
) {

    val nodeList = mutableListOf<ComputationGraph.Node>()

    fun floatVariable(
        name: String,
        shape: IntArray1D = intArray1DOf()
    ): Variable = Variable(
        scope,
        shape,
        DataType.FLOAT
    ).also { nodeList += it }

    fun floatConstant(
        value: DoubleArrayND
    ): RealConstant = RealConstant(value, DataType.FLOAT).also { nodeList += it }

    operator fun ComputationGraph.Expression.plus(
        other: ComputationGraph.Expression
    ): Add = Add(this, other).also { nodeList += it }

    fun build() = ComputationGraph(nodeList)

}


interface NodeCompiler<T: ComputationGraph.Node> {

    fun compile(
        context: TensorflowCompiler.Context,
        node: T
    ): Operand<*>

}


class TensorflowAddCompiler: NodeCompiler<Add> {

    override fun compile(
        context: TensorflowCompiler.Context,
        node: Add
    ): Operand<*> {

        val opBuilder = context.graph.opBuilder("Add", context.scope.makeOpName("Add")).apply {
            addInput(context.compile(node.input1).asOutput())
            addInput(context.compile(node.input2).asOutput())
        }
        return opBuilder.build().output<Any>(0)

    }

}



class TensorflowCompiler {

    private val classCompilerMap = mutableMapOf<Class<*>, NodeCompiler<*>>()

    fun <T: ComputationGraph.Node> register(
        nodeType: Class<T>,
        compiler: NodeCompiler<T>
    ) {
        classCompilerMap[nodeType] = compiler
    }

    inline fun <reified T: ComputationGraph.Node> register(
        compiler: NodeCompiler<T>
    ) {
        register(T::class.java, compiler)
    }

    fun searchCompilerForNode(
        node: ComputationGraph.Node
    ): NodeCompiler<ComputationGraph.Node> =
        classCompilerMap[node::class.java] as? NodeCompiler<ComputationGraph.Node> ?:
        error("No compiler for node $node")

    class Context(
        private val compiler: TensorflowCompiler,
        val bindings: Map<ComputationGraph.Node, Operand<*>>,
        val graph: Graph,
        val scope: Scope
    ) {

        fun compile(node: ComputationGraph.Node): Operand<*> =
            compiler.compile(this, node)


    }

    fun compile(
        context: Context,
        node: ComputationGraph.Node
    ): Operand<*> =
        searchCompilerForNode(node).compile(context, node)

    fun compile(graph: ComputationGraph): Graph {

        val bindings = mutableMapOf<ComputationGraph.Node, Operand<*>>()

        val result = Graph()

        val context = Context(
            compiler = this,
            bindings = bindings,
            graph = Graph(),
            scope = Scope(result)
        )

        for (node in graph.nodeList) {

            bindings[node] = compile(
                context,
                node
            )

        }

        return result
    }

}


*/