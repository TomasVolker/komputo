package tomasvolker.komputo.dsl.builder

import org.tensorflow.*
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.*
import org.tensorflow.op.core.Identity
import org.tensorflow.op.core.Relu
import org.tensorflow.op.core.Sigmoid
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.asOfNumber
import tomasvolker.komputo.dsl.*
import tomasvolker.komputo.toClass
import tomasvolker.komputo.toShape
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.intArray1D
import tomasvolker.numeriko.core.interfaces.factory.intArray1DOf

open class Model(
    val graph: Graph,
    val inputList: List<Operand<*>>,
    val outputList: List<Operand<*>>,
    val initializeOperation: Operand<*>? = null
) {

    val inputSize: Int = inputList.size
    val outputSize: Int = outputList.size

}



open class ModelBuilder(
    val ops: Ops,
    val initializationList: MutableList<Operand<*>> = mutableListOf(),
    val inputList: MutableList<Placeholder<*>> = mutableListOf(),
    val outputList: MutableList<Operand<*>> = mutableListOf(),
    val targetList: MutableList<Placeholder<*>> = mutableListOf(),
    val trainableVariableList: MutableList<Variable<*>> = mutableListOf()
) {

    var defaultDataType: DataType = DataType.FLOAT
    var defaultFloatDataType: DataType = DataType.FLOAT
    var defaultIntegerDataType: DataType = DataType.INT32

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
        data: Int,
        dataType: DataType = defaultIntegerDataType
    ): Constant<*> = ops.constant(data, dataType)



    fun output(
        operand: Operand<*>,
        name: String? = null,
        trainable: Boolean = true
    ): Operand<*> {

        val result = if(name != null)
            ops.withNameOrSame(name).identity(operand)
        else
            operand

        outputList += result
        if (trainable)
            targetList += placeholder("${result.localName}_target", result.shape, operand.dataType)

        return result
    }


    fun assign(
        variable: Variable<*>,
        value: Operand<*>,
        name: String? = null
    ): Assign<*> = ops.assign(variable, value, name)

    infix fun Variable<*>.assignTo(value: Operand<*>): Assign<*> = assign(this, value)



    fun variable(
        name: String? = null,
        initialValue: Operand<*>? = null,
        shape: IntArray1D? = null,
        dataType: DataType = initialValue?.dataType ?: defaultDataType,
        trainable: Boolean = true
    ): Variable<*> = ops.variable(dataType, name, shape).also { variable ->
        if (trainable) {
            trainableVariableList.add(variable)
        }

        initialValue?.let {
            initializationList += assign(variable, it, name = "init_${variable.name.substringAfter('/')}")
        } ?: Unit
    }



    inline fun <T> scope(name: String, init: ModelBuilder.()->T): T =
        ModelBuilder(
            ops.withSubScope(name),
            initializationList = initializationList,
            inputList = inputList,
            outputList = outputList,
            targetList = targetList,
            trainableVariableList = trainableVariableList
        ).run(init)


    operator fun Operand<*>.unaryPlus(): Identity<*> = ops.identity(this)
    operator fun Operand<*>.unaryMinus(): Neg<*> = ops.neg(this)

    operator fun Operand<*>.plus(other: Operand<*>): Add<*> = ops.add(this.asOfAny(), other.asOfAny())
    operator fun Operand<*>.minus(other: Operand<*>): Sub<*> = ops.sub(this.asOfAny(), other.asOfAny())
    operator fun Operand<*>.times(other: Operand<*>): Mul<*> = ops.mul(this.asOfAny(), other.asOfAny())
    operator fun Operand<*>.div(other: Operand<*>): Div<*> = ops.div(this.asOfAny(), other.asOfAny())

    infix fun Operand<*>.matmul(other: Operand<*>): MatMul<*> = ops.matMul(this.asOfAny(), other.asOfAny())


    fun randomNormal(
        shape: IntArray1D = intArray1DOf(),
        dataType: DataType = defaultFloatDataType
    ): RandomNormal<*> =
        ops.randomNormal(constant(shape).asOfNumber(), dataType.toClass() as Class<Number>)


    fun randomNormal(
        shape: IntArray1D,
        mean: Double = 0.0,
        deviation: Double = 1.0,
        dataType: DataType = defaultFloatDataType
    ): Operand<*> {

        var result: Operand<*> = randomNormal(shape, dataType)

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

    fun identity(input: Operand<*>): Operand<*> = ops.identity(input)

    fun sigmoid(input: Operand<*>): Sigmoid<*> = ops.sigmoid(input)
    fun softmax(input: Operand<*>): Softmax<*> = ops.softmax(input.asOfNumber())
    fun relu(input: Operand<*>): Relu<*> = ops.relu(input)

    fun softmaxCrossEntropyWithLogits(
        output: Operand<*>,
        target: Operand<*>
    ): Operand<*> = ops.softmaxCrossEntropyWithLogits(
        output.asOfNumber(),
        target.asOfNumber()
    ).loss().asOutput()

    fun square(input: Operand<*>): Square<*> = ops.square(input)
    fun Operand<*>.squared(): Square<*> = square(this)

    fun broadcastTo(input: Operand<*>, shape: Operand<*>): BroadcastTo<*> =
        ops.broadcastTo(input.asOfAny(), shape.asOfNumber())

    fun mean(input: Operand<*>, axis: Operand<*>): Mean<*> = ops.mean(input, axis.asOfNumber())
    fun mean(input: Operand<*>, axis: Int): Mean<*> = ops.mean(input, constant(axis).asOfNumber())

    fun reduceMean(input: Operand<*>, axis: Operand<*>): ReduceMean<*> = ops.reduceMean(input, axis.asOfNumber())
    fun reduceMean(input: Operand<*>, axis: IntArray1D): ReduceMean<*> = ops.reduceMean(input, constant(axis).asOfNumber())

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


inline fun Graph.withBuilder(init: ModelBuilder.()->Unit) = this.also { ModelBuilder(it).init() }

inline fun model(init: ModelBuilder.()->Unit) = ModelBuilder(Graph()).apply(init).build()


