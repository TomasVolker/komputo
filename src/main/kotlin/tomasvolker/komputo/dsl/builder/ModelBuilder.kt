package tomasvolker.komputo.dsl.builder

import org.tensorflow.*
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.*
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.factory.intArray1D
import tomasvolker.numeriko.core.interfaces.factory.intArray1DOf
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D
import tomasvolker.numeriko.core.operations.concatenate
import tomasvolker.tensorflow.dsl.*

open class Model(
    val graph: Graph,
    val inputList: List<Placeholder<*>>,
    val outputList: List<Operand<*>>,
    val initializeOperation: Operand<*>? = null
) {

    val inputSize: Int = inputList.size
    val outputSize: Int = outputList.size

}


fun Shape.toIntArray1D() = intArray1D(numDimensions()) { i -> size(i).toInt() }

open class ModelBuilder(
    val ops: Ops,
    val initializationList: MutableList<Operand<*>> = mutableListOf(),
    val inputList: MutableList<Placeholder<*>> = mutableListOf(),
    val outputList: MutableList<Operand<*>> = mutableListOf(),
    val targetList: MutableList<Placeholder<*>> = mutableListOf(),
    val trainableVariableList: MutableList<Variable<Float>> = mutableListOf()
) {

    val scope: Scope = ops.scope()
    val graph: Graph = scope.graph()

    constructor(scope: Scope): this(scope.graph())
    constructor(graph: Graph): this(Ops.create(graph))

    // Internal variables

    var output: Operand<*>
        get() = outputList.first()
        set(value) {
            outputList.clear()
            outputList += value
        }

    var target: Placeholder<*>
        get() = targetList.first()
        set(value) {
            targetList.clear()
            targetList += value
        }

    fun sequential(input: Operand<*>, sequence: SequentialBuilder.()->Unit): Operand<*> =
        SequentialBuilder(this, input).apply(sequence).lastOutput

    open fun build(): Model = Model(
        graph = graph,
        inputList = inputList,
        outputList = outputList,
        initializeOperation = group("initialize_variables", initializationList)
    )

    val lastIndex: Int get() = -1
    val dynamic: Int get() = -1

    inline fun <reified T> variable(
        name: String? = null,
        initialValue: Operand<T>? = null,
        shape: IntArray1D? = null,
        trainable: Boolean = true
    ): Variable<T> = ops.variable<T>(name, shape).also { variable ->
        if (T::class.java == java.lang.Float::class.java && trainable) {
            trainableVariableList.add(variable as Variable<Float>)
        }

        initialValue?.let {
            initializationList += assign(variable, it, name = "init_${variable.name.substringAfter('/')}")
        }
    }

    fun constant(data: IntArray1D): Constant<Int> = ops.constant(data.toIntArray())

    inline fun <reified T: Number> constant(data: Double): Constant<T> = ops.constant(data.cast<T>(), T::class.java)

    fun constant(data: Float): Constant<Float> = ops.constant(data)
    fun constant(data: Int): Constant<Int> = ops.constant(data)

    inline fun <reified T> constant(data: T): Constant<T> = ops.constant(data, T::class.java)

    fun constant(data: DoubleArray1D): Constant<Double> = ops.constant(data.toDoubleArray())

    inline fun <reified T> placeholder(
        name: String? = null,
        shape: IntArray1D? = null
    ): Placeholder<T> = ops.placeholder(name, shape)

    inline fun <reified T> placeholder(
        name: String? = null,
        shape: Shape
    ): Placeholder<T> = ops.placeholder(name, shape)

    inline fun <reified T> input(
        name: String? = null,
        shape: IntArray1D? = null
    ): Placeholder<T> = placeholder<T>(name, shape).also {
        inputList += it
    }

    inline fun <reified T> output(
        operand: Operand<*>,
        name: String? = null,
        trainable: Boolean = true
    ): Operand<T> {
        val casted = operand as Operand<T>
        return (name?.let { ops.identity(casted) } ?: operand).also {
            outputList += it
            if (trainable) targetList += placeholder<T>("target_${it.name}", it.shape)
        }
    }

    inline fun <reified T> assign(
        variable: Operand<T>,
        value: Operand<T>,
        name: String? = null
    ): Assign<T> = ops.assign(variable, value, name)

    inline infix fun <reified T> Variable<T>.assignTo(value: Operand<T>): Assign<T> = assign(this, value)

    inline fun <T> scope(name: String, init: ModelBuilder.()->T): T =
        ModelBuilder(
            ops.withSubScope(name),
            initializationList = initializationList,
            inputList = inputList,
            outputList = outputList,
            targetList = targetList,
            trainableVariableList = trainableVariableList
        ).run(init)

    operator fun <T> Operand<T>.unaryPlus(): Identity<T> = ops.identity(this)
    operator fun <T> Operand<T>.unaryMinus(): Neg<T> = ops.neg(this)

    operator fun <T> Operand<T>.plus(other: Operand<T>): Add<T> = ops.add(this, other)
    operator fun <T> Operand<T>.minus(other: Operand<T>): Sub<T> = ops.sub(this, other)
    operator fun <T> Operand<T>.times(other: Operand<T>): Mul<T> = ops.mul(this, other)
    operator fun <T> Operand<T>.div(other: Operand<T>): Div<T> = ops.div(this, other)

    infix fun <T> Operand<T>.matmul(other: Operand<T>): MatMul<T> = ops.matMul(this, other)

    inline fun <reified T: Number> randomNormal(shape: IntArray1D = intArray1DOf()): RandomNormal<T> =
        ops.randomNormal(constant(shape), T::class.java)

    inline fun <reified T: Number> randomNormal(
        shape: IntArray1D,
        mean: Double = 0.0,
        deviation: Double = 1.0
    ): Operand<T> {

        var result: Operand<T> = randomNormal<T>(shape)

        scope("RandomNormal") {

            if (deviation != 1.0) {
                result *= constant(deviation)
            }

            if (mean != 0.0) {
                result += constant(mean)
            }

        }

        return result
    }

    // Activations

    fun <T> sigmoid(input: Operand<T>): Sigmoid<T> = ops.sigmoid(input)
    fun <T: Number> softmax(input: Operand<T>): Softmax<T> = ops.softmax(input)
    fun <T: Number> relu(input: Operand<T>): Relu<T> = ops.relu(input)

    fun <T> square(input: Operand<T>): Square<T> = ops.square(input)
    fun <T> Operand<T>.squared(): Square<T> = square(this)

    fun <T, U: Number> mean(input: Operand<T>, axis: Operand<U>): Mean<T> = ops.mean(input, axis)
    fun <T> mean(input: Operand<T>, axis: Int): Mean<T> = ops.mean(input, constant(axis))

    fun group(operations: Iterable<Operand<*>>): Operand<*> = ops.group(operations)
    fun group(vararg operations: Operand<*>): Operand<*> = ops.group(*operations)

    fun group(name: String, vararg operations: Operand<*>): Operand<*> = ops.group(name, *operations)
    fun group(name: String, operations: Iterable<Operand<*>>): Operand<*> = ops.group(name, operations)

    fun gradients(value: Operand<*>, vararg wrt: Operand<*>): Gradients = gradients(value, wrt.toList())
    fun gradients(value: List<Operand<*>>, vararg wrt: Operand<*>): Gradients = gradients(value, wrt.toList())
    fun gradients(value: Operand<*>, wrtList: List<Operand<*>>, name: String? = null): Gradients =
        ops.withNameOrSame(name).gradients(value, wrtList)
    fun gradients(value: List<Operand<*>>, wrtList: List<Operand<*>>, name: String? = null): Gradients =
        ops.withNameOrSame(name).gradients(value, wrtList)

}

inline fun <reified T: Number> Number.cast(): T = when(val clazz = T::class.java) {
    java.lang.Double::class.java -> toDouble()
    java.lang.Float::class.java -> toFloat()
    java.lang.Long::class.java -> toLong()
    java.lang.Integer::class.java -> toInt()
    java.lang.Short::class.java -> toShort()
    java.lang.Character::class.java -> toChar()
    else -> error(clazz.name)
} as T

inline fun Graph.withBuilder(init: ModelBuilder.()->Unit) = this.also { ModelBuilder(it).init() }

inline fun model(init: ModelBuilder.()->Unit) = ModelBuilder(Graph()).apply(init).build()


