package tomasvolker.tensorflow.dsl

import org.tensorflow.Operand
import org.tensorflow.Operation
import org.tensorflow.OperationBuilder
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.*
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D


inline fun Ops.scope(name: String, init: Ops.()->Unit): Ops =
    withSubScope(name).apply(init)

fun Ops.withNameOrSame(name: String? = null) = name?.let { withName(it) } ?: this

val Ops.lastIndex: Int get() = -1
val Ops.dynamic: Int get() = -1

inline fun <reified T> Ops.placeholder(
    name: String? = null,
    shape: Shape = Shape.unknown(),
    dtype: Class<T> = T::class.java
): Placeholder<T> =
    withNameOrSame(name).placeholder(dtype, Placeholder.shape(shape))

inline fun <reified T> Ops.placeholder(
    name: String? = null,
    shape: IntArray1D? = null
): Placeholder<T> = placeholder(name, shape?.toShape() ?: Shape.unknown())

fun IntArray1D.toShape(): Shape =
    Shape.make(
        this[0].toLong(),
        *this.drop(1).map { it.toLong() }.toLongArray()
    )

inline fun <reified T> Ops.variable(
    name: String? = null,
    shape: Shape = Shape.unknown(),
    dtype: Class<T> = T::class.java
): Variable<T> =
    withNameOrSame(name).variable(shape, dtype)

inline fun <reified T> Ops.variable(
    name: String? = null,
    shape: IntArray1D? = null
): Variable<T> = variable(name, shape?.toShape() ?: Shape.unknown())

inline fun <reified T> Ops.assign(
    variable: Operand<T>,
    value: Operand<T>,
    name: String? = null,
    dtype: Class<T> = T::class.java
): Assign<T> =
    withNameOrSame(name).assign(variable, value)


fun Ops.gradientDescent(
    cost: Operand<Float>,
    variableList: List<Variable<Float>>,
    rate: Double
): Operand<*> {

    var result: List<Operand<*>>? = null

    scope("gradient_descent") {

        val grad = withName("gradient").gradients(cost, variableList)

        val alpha = constant(rate.toFloat())

        result = variableList.mapIndexed { i, variable ->
            applyGradientDescent(variable, alpha, grad.dy(i))
        }

    }

    return withName("Train").group(result ?: error(""))
}

inline fun Ops.buildOp(
    operation: String,
    name: String = scope().makeOpName(operation),
    init: OperationBuilder.()->Unit = {}
) =
    scope().graph().opBuilder(operation, name).apply(init).build()

fun Ops.constant(shape: IntArray1D): Constant<Int> = constant(shape.toIntArray())

inline fun <reified T: Number> Ops.randomNormal(shape: IntArray1D): RandomNormal<T> =
    randomNormal(constant(shape), T::class.java)

fun <T> Ops.reshape(operand: Operand<T>, shape: IntArray1D): Reshape<T> =
    reshape(operand, constant(shape.toIntArray()))

fun Ops.group(vararg operands: Operand<*>): Operand<*> =
    group(operands.toList())

fun Ops.group(operands: Iterable<Operand<*>>): Operand<*> =
        group("Group", operands)

fun Ops.group(name: String, vararg operands: Operand<*>): Operand<*> =
        group(name, operands.toList())

fun Ops.group(name: String, operands: Iterable<Operand<*>>): Operand<*> =
    buildOp("NoOp", name = scope().makeOpName(name)) {
        operands.forEach {
            addControlInput(it.asOutput().op())
        }
    }.output<Any>(0)

fun Ops.noOperation(name: String? = null): Operand<*> =
    buildOp("NoOp", name = scope().makeOpName(name)).output<Any>(0)

