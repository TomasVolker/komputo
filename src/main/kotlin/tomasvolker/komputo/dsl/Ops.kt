package tomasvolker.komputo.dsl

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.OperationBuilder
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.*
import tomasvolker.komputo.*
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND


inline fun Ops.scope(name: String, init: Ops.()->Unit): Ops =
    withSubScope(name).apply(init)

fun Ops.withNameOrSame(name: String? = null) = name?.let { withName(it) } ?: this


val Ops.lastIndex: Int get() = -1
val Ops.dynamic: Int get() = -1


fun Ops.placeholder(
    dtype: DataType,
    name: String? = null,
    shape: IntArray1D? = null
): Placeholder<*> =
    withNameOrSame(name).placeholder(dtype.toClass(), Placeholder.shape(shape.toShape()))


fun Ops.variable(
    dataType: DataType,
    name: String? = null,
    shape: IntArray1D? = null
): Variable<*> = withNameOrSame(name).variable(shape.toShape(), dataType.toClass())



fun Ops.assign(
    variable: Variable<*>,
    value: Operand<*>,
    name: String? = null
): Assign<*> =
    withNameOrSame(name).assign(variable.asOfAny(), value.asOfAny())



fun Ops.constant(
    value: DoubleArrayND,
    dataType: DataType = DataType.FLOAT,
    name: String? = null
): Constant<*> =
    value.toTensor().use {
        withNameOrSame(name).constant(it, dataType.toClass()) // cast value
    }


fun Ops.constant(
    value: IntArray1D,
    dataType: DataType = DataType.INT32,
    name: String? = null
): Constant<*> =
    withNameOrSame(name).constant(value.toIntArray(), dataType.toClass()) // cast value


fun Ops.constant(
    value: Double,
    dataType: DataType,
    name: String? = null
): Constant<*> =
    withNameOrSame(name).constant(value.castTo(dataType), dataType.toClass())


fun Ops.constant(
    value: Int,
    dataType: DataType,
    name: String? = null
): Constant<*> =
    withNameOrSame(name).constant(value.castTo(dataType), dataType.toClass())



fun Ops.gradientDescent(
    cost: Operand<*>,
    variableList: List<Variable<*>>,
    rate: Double,
    dataType: DataType
): Operand<*> {

    var result: List<Operand<*>>? = null

    scope("gradient_descent") {

        val grad = withName("gradient").gradients(cost, variableList)

        val alpha = constant(rate, dataType)

        result = variableList.mapIndexed { i, variable ->
            applyGradientDescent<Any>(variable.asOfAny(), alpha.asOfAny(), grad.dy(i))
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



fun Ops.randomNormal(
    shape: IntArray1D,
    dataType: DataType
): RandomNormal<*> =
    randomNormal<Number, Number>(
        constant(shape, dataType).asOfNumber(),
        dataType.toClass() as Class<Number>
    )


fun Ops.reshape(operand: Operand<*>, shape: IntArray1D): Reshape<*> =
    reshape(operand.asOfAny(), constant(shape).asOfNumber())


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

