package tomasvolker.komputo.builder

import org.tensorflow.*
import org.tensorflow.op.Operands
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.*
import org.tensorflow.op.core.Identity
import org.tensorflow.op.core.Relu
import org.tensorflow.op.core.Sigmoid
import tomasvolker.komputo.*
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.intArray1DOf
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D
import tomasvolker.numeriko.core.operations.concatenate
import java.io.File
import java.nio.charset.Charset

fun Model.saveGraphDef(filename: String) {
    File(filename).writeBytes(graph.toGraphDef())
}

open class Model(
    val builder: ModelBuilder,
    val inputList: List<TFOperand>,
    val outputList: List<TFOperand>,
    val parameterList: List<TFVariable>,
    val variableList: List<TFVariable>,
    val regularizationList: List<TFOperand> = mutableListOf(),
    val trainingFactor: PlaceholderWithDefault<*>,
    val initializeList: List<TFOperand> = emptyList()
) {

    val inputSize: Int = inputList.size
    val outputSize: Int = outputList.size

    val graph get() = builder.graph

}


open class ModelBuilder(
    val ops: Ops,
    val initializationList: MutableList<TFOperand> = mutableListOf(),
    val inputList: MutableList<Placeholder<*>> = mutableListOf(),
    val outputList: MutableList<TFOperand> = mutableListOf(),
    val parameterList: MutableList<Variable<*>> = mutableListOf(),
    val variableList: MutableList<TFVariable> = mutableListOf(),
    val regularizationList: MutableList<TFOperand> = mutableListOf(),
    val trainingFactor: PlaceholderWithDefault<*> =
        ops.placeholderWithDefault(ops.constant(0.0f), intArray1DOf(), name = "training_factor")
) {

    var defaultDataType: DataType = DataType.FLOAT
    var defaultFloatDataType: DataType = DataType.FLOAT
    var defaultIntegerDataType: DataType = DataType.INT32

    val scope: Scope = ops.scope()
    val graph: Graph = scope.graph()

    constructor(scope: Scope): this(scope.graph())
    constructor(graph: Graph): this(Ops.create(graph))

    // Internal variables

    var output: TFOperand
        get() = outputList.first()
        set(value) {
            outputList.clear()
            outputList += value
        }

    inline fun sequential(input: TFOperand, sequence: SequentialBuilder.()->Unit): TFOperand =
        SequentialBuilder(this, input).apply(sequence).output

    open fun build(): Model = Model(
        builder = this,
        inputList = inputList,
        outputList = outputList,
        initializeList = initializationList,
        regularizationList = regularizationList,
        parameterList = parameterList,
        variableList = variableList,
        trainingFactor = trainingFactor
    )

    val lastIndex: Int get() = -1
    val dynamic: Int get() = -1


    fun placeholderWithDefault(
        default: TFOperand,
        shape: IntArray1D? = null,
        name: String? = null
    ): PlaceholderWithDefault<*> =
        ops.placeholderWithDefault(default, shape, name)



    fun placeholder(
        name: String? = null,
        shape: IntArray1D? = null,
        dataType: DataType = defaultDataType
    ): Placeholder<*> = ops.placeholder(dataType, name, shape)


    fun input(
        name: String? = null,
        shape: IntArray1D? = null
    ): Placeholder<*> = placeholder(name, shape).also {
        inputList += it
    }


    fun constant(
        data: Double,
        dataType: DataType = defaultFloatDataType
    ): Constant<*> = ops.constant(data, dataType)

    fun constant(
        data: IntArray1D,
        dataType: DataType = defaultIntegerDataType
    ): Constant<*> = ops.constant(data, dataType)

    fun constant(
        data: DoubleArrayND,
        dataType: DataType = defaultFloatDataType
    ): Constant<*> = ops.constant(data, dataType)

    fun constant(
        data: List<String>
    ): Constant<*> = ops.constant(data.map { it.toByteArray(Charsets.US_ASCII) }.toTypedArray(), DataType.STRING.toClass())

    fun constant(
        data: String
    ): Constant<*> = ops.constant(data.toByteArray(Charsets.US_ASCII), DataType.STRING.toClass())

    fun constant(
        data: Int,
        dataType: DataType = defaultIntegerDataType
    ): Constant<*> = ops.constant(data, dataType)



    fun output(
        operand: TFOperand,
        name: String? = null
    ): TFOperand {

        val result = if(name != null)
            ops.withNameOrSame(name).identity(operand)
        else
            operand

        outputList += result

        return result
    }


    fun assign(
        variable: Variable<*>,
        value: TFOperand,
        name: String? = null
    ): Assign<*> = ops.assign(variable, value, name)

    infix fun Variable<*>.assignTo(value: TFOperand): Assign<*> = assign(this, value)



    fun variable(
        name: String? = null,
        initialValue: TFOperand? = null,
        shape: IntArray1D? = null,
        dataType: DataType = initialValue?.dataType ?: defaultDataType
    ): Variable<*> = ops.variable(dataType, name, shape).also { variable ->
        variableList += variable
        initialValue?.let {
            initializationList += assign(variable, it, name = "${variable.localName}_init")
        } ?: Unit
    }

    fun parameter(
        name: String? = null,
        initialValue: TFOperand? = null,
        shape: IntArray1D? = null,
        dataType: DataType = initialValue?.dataType ?: defaultDataType,
        trainable: Boolean = true
    ): Variable<*> = variable(name, initialValue, shape, dataType).also { variable ->
        if (trainable) {
            parameterList.add(variable)
        }
    }

    fun regularize(operand: TFOperand) {
        regularizationList += operand
    }

    inline fun <T> scope(name: String, init: ModelBuilder.()->T): T =
        ModelBuilder(
            ops.withSubScope(name),
            initializationList = initializationList,
            inputList = inputList,
            outputList = outputList,
            parameterList = parameterList,
            variableList = variableList,
            regularizationList = regularizationList,
            trainingFactor = trainingFactor
        ).run(init)


    operator fun TFOperand.unaryPlus(): Identity<*> = ops.identity(this)
    operator fun TFOperand.unaryMinus(): Neg<*> = ops.neg(this)

    operator fun TFOperand.plus(other: TFOperand): Add<*> = ops.add(this.asOfAny(), other.asOfAny())
    operator fun TFOperand.minus(other: TFOperand): Sub<*> = ops.sub(this.asOfAny(), other.asOfAny())
    operator fun TFOperand.times(other: TFOperand): Mul<*> = ops.mul(this.asOfAny(), other.asOfAny())
    operator fun TFOperand.div(other: TFOperand): Div<*> = ops.div(this.asOfAny(), other.asOfAny())

    operator fun Int.plus(other: TFOperand) = constant(this) + other
    operator fun Int.minus(other: TFOperand) = constant(this) - other
    operator fun Int.times(other: TFOperand) = constant(this) * other
    operator fun Int.div(other: TFOperand) = constant(this) / other

    operator fun TFOperand.plus(other: Int) = this + constant(other)
    operator fun TFOperand.minus(other: Int) = this - constant(other)
    operator fun TFOperand.times(other: Int) = this * constant(other)
    operator fun TFOperand.div(other: Int) = this / constant(other)

    operator fun Double.plus(other: TFOperand) = constant(this) + other
    operator fun Double.minus(other: TFOperand) = constant(this) - other
    operator fun Double.times(other: TFOperand) = constant(this) * other
    operator fun Double.div(other: TFOperand) = constant(this) / other

    operator fun TFOperand.plus(other: Double) = this + constant(other)
    operator fun TFOperand.minus(other: Double) = this - constant(other)
    operator fun TFOperand.times(other: Double) = this * constant(other)
    operator fun TFOperand.div(other: Double) = this / constant(other)

    infix fun TFOperand.matmul(other: TFOperand): MatMul<*> = ops.matMul(this.asOfAny(), other.asOfAny())


    fun randomNormal(
        shape: IntArray1D = intArray1DOf(),
        dataType: DataType = defaultFloatDataType
    ): RandomNormal<*> =
        ops.randomNormal(shape, dataType)


    fun randomUniform(
        shape: IntArray1D = intArray1DOf(),
        dataType: DataType = defaultFloatDataType
    ): RandomUniform<*> =
        ops.randomUniform(shape, dataType)


    fun randomNormal(
        shape: IntArray1D,
        mean: Double = 0.0,
        deviation: Double = 1.0,
        dataType: DataType = defaultFloatDataType
    ): TFOperand {

        var result: TFOperand = randomNormal(shape, dataType)

        scope("RandomNormal") {

            if (deviation != 1.0) {
                result *= constant(deviation, dataType)
            }

            if (mean != 0.0) {
                result += constant(mean, dataType)
            }

        }

        return result
    }

    // Activations

    fun identity(input: TFOperand): TFOperand = ops.identity(input)

    fun sigmoid(input: TFOperand): Sigmoid<*> = ops.sigmoid(input)
    fun softmax(input: TFOperand): Softmax<*> = ops.softmax(input.asOfNumber())
    fun relu(input: TFOperand): Relu<*> = ops.relu(input)

    fun softmaxCrossEntropyWithLogits(
        output: TFOperand,
        target: TFOperand
    ): TFOperand = ops.softmaxCrossEntropyWithLogits(
        output.asOfNumber(),
        target.asOfNumber()
    ).loss().asOutput()

    fun square(input: TFOperand): Square<*> = ops.square(input)
    fun TFOperand.squared(): Square<*> = square(this)

    fun abs(input: TFOperand): Abs<*> = ops.abs(input.asOfNumber())

    fun floor(input: TFOperand): Floor<*> = ops.floor(input.asOfNumber())

    fun broadcastTo(input: TFOperand, shape: TFOperand): BroadcastTo<*> =
        ops.broadcastTo(input.asOfAny(), shape.asOfNumber())

    fun mean(input: TFOperand, axis: TFOperand): Mean<*> = ops.mean(input, axis.asOfNumber())
    fun mean(input: TFOperand, axis: Int): Mean<*> = ops.mean(input, constant(axis).asOfNumber())

    fun sum(input: TFOperand, axis: Int): Sum<*> = ops.sum(input, constant(axis).asOfNumber())
    fun sum(input: TFOperand, axis: IntArray1D): Sum<*> = ops.sum(input, constant(axis).asOfNumber())

    fun reduceMean(input: TFOperand, axis: TFOperand): ReduceMean<*> = ops.reduceMean(input, axis.asOfNumber())
    fun reduceMean(input: TFOperand, axis: IntArray1D): ReduceMean<*> = ops.reduceMean(input, constant(axis).asOfNumber())

    fun gradients(value: TFOperand, vararg wrt: TFOperand): Gradients = gradients(value, wrt.toList())
    fun gradients(value: List<TFOperand>, vararg wrt: TFOperand): Gradients = gradients(value, wrt.toList())

    fun gradients(value: TFOperand, wrtList: List<TFOperand>, name: String? = null): Gradients =
        ops.withNameOrSame(name).gradients(value, wrtList)

    fun gradients(value: List<TFOperand>, wrtList: List<TFOperand>, name: String? = null): Gradients =
        ops.withNameOrSame(name).gradients(value, wrtList)

}


inline fun Graph.withBuilder(init: ModelBuilder.()->Unit) = this.also { ModelBuilder(it).init() }

inline fun graphModel(init: ModelBuilder.()->Unit) = ModelBuilder(Graph()).apply(init).build()

inline fun sequentialModel(inputShape: IntArray1D, init: SequentialBuilder.()->Unit) = graphModel {
    output(
        sequential(
            input(shape = I[dynamic] concatenate inputShape),
            init
        )
    )
}

inline fun sequentialModel(vararg inputShape: Int, init: SequentialBuilder.()->Unit) =
    sequentialModel(inputShape.toIntArray1D(), init)


