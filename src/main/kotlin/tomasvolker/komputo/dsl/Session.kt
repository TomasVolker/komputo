package tomasvolker.tensorflow.dsl

import org.tensorflow.*
import tomasvolker.komputo.dsl.builder.Model
import tomasvolker.komputo.dsl.builder.TrainableModel
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND

class TrainableModelSession(
    override val model: TrainableModel
): ModelSession(model) {

    fun Model.fit(inputs: DoubleArrayND, targets: DoubleArrayND): List<DoubleArrayND> =
            fit(listOf(inputs), listOf(targets))

    fun Model.fit(inputs: List<DoubleArrayND>, targets: List<DoubleArrayND>): List<DoubleArrayND> {

        val inputList = inputs.map { it.toTensor() }
        val targetList = targets.map { it.toTensor() }

        try {

            val resultTensors =  model.fit(inputList, targetList)

            try {
                return resultTensors.map { it.toDoubleNDArray() }
            } finally {
                resultTensors.forEach { it.close() }
            }

        } finally {
            inputList.forEach { it.close() }
            targetList.forEach { it.close() }
        }

    }

    @JvmName("tensorFit")
    fun Model.fit(inputs: List<Tensor<*>>, targets: List<Tensor<*>>): List<Tensor<*>> {

        require(inputs.size == inputSize) {
            "${inputs.size} inputs were provided when $inputSize are needed"
        }

        require(targets.size == outputSize) {
            "${targets.size} targets were provided when $inputSize are needed"
        }

        return with(tensorflowSession) {
            compute {
                for (i in 0 until inputSize) {
                    feed(model.inputList[i].output(), inputs[i])
                    feed(model.targetList[i].output(), targets[i])
                    fetch(model.loss)
                }
                model.trainOperation?.let { addTarget(it) }
            }
        }
    }

}

open class ModelSession(
    model: Model
): AutoCloseable {

    val tensorflowSession = Session(model.graph)

    open val model: Model = model

    fun initialize() {
        model.initialize()
    }

    fun Model.initialize() {
        model.initializeOperation?.let { tensorflowSession.execute(it) }
    }

    operator fun Model.invoke(vararg inputs: DoubleArrayND): List<DoubleArrayND> =
        invoke(inputs.toList())

    operator fun Model.invoke(inputs: List<DoubleArrayND>): List<DoubleArrayND> {
        val tensorList = inputs.map { it.toTensor() }
        try {

            val tensorResults = invoke(tensorList)

            try {
                return tensorResults.map { it.toDoubleNDArray() }
            } finally {
                tensorResults.forEach { it.close() }
            }

        } finally {
            tensorList.forEach { it.close() }
        }
    }

    fun compute(vararg inputs: DoubleArrayND): List<DoubleArrayND> = model(*inputs)

    @JvmName("tensorInvoke")
    operator fun Model.invoke(inputs: List<Tensor<*>>): List<Tensor<*>> {

        require(inputs.size == inputSize) {
            "${inputs.size} were provided when $inputSize are needed"
        }

        return with(tensorflowSession) {
            compute {
                inputList.zip(inputs) { operand, input ->
                    feed(operand.output(), input)
                }
                fetchAll(outputList)
            }
        }
    }

    override fun close() {
        tensorflowSession.close()
    }

}

inline fun <T> session(graph: Model, block: ModelSession.()->T) = ModelSession(graph).use(block)

inline fun <T> trainSession(graph: TrainableModel, block: TrainableModelSession.()->T) =
    TrainableModelSession(graph).use(block)

inline fun <T> session(graph: Graph, block: Session.()->T) = Session(graph).use(block)

fun Session.compute(init: Session.Runner.()->Unit): List<Tensor<*>> =
    runner().apply(init).run()

fun Session.execute(operand: Operand<*>?): List<Tensor<*>> =
    runner().apply { addTarget(operand) }.run()

fun Session.compute(operand: Operand<*>, feedMap: Map<Output<*>, Tensor<*>>): List<Tensor<*>> =
    compute {
        feedMap.forEach { operand, tensor ->
            feed(operand, tensor)
        }
        fetch(operand)
    }

fun Session.execute(operand: Operand<*>, feedMap: Map<Output<*>, Tensor<*>>) {
    compute {
        feedMap.forEach { operand, tensor ->
            feed(operand, tensor)
        }
        addTarget(operand)
    }
}

fun Session.Runner.fetchAll(operations: Iterable<Operand<*>>): Session.Runner =
    apply { operations.forEach { fetch(it) } }
