package tomasvolker.komputo.builder

import org.tensorflow.Operand
import org.tensorflow.Operation
import org.tensorflow.Session
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.arraynd.generic.ArrayND
import tomasvolker.numeriko.core.interfaces.factory.array0D

fun ModelSession<TrainableModel>.save(filename: String) {
    execute(model.save, feed = mapOf(model.filename to array0D(filename)))
}

fun ModelSession<TrainableModel>.restore(filename: String) {
    execute(model.restore, feed = mapOf(model.filename to array0D(filename)))
}

fun <M: Model> session(model: M, initialize: Boolean = true, block: ModelSession<M>.()->Unit) {
    ModelSession(model).use { session ->

        if (initialize) {
            model.initializeList.forEach {
                session.execute(it)
            }
        }

        session.block()

    }
}

class ModelSession<out M: Model>(
    val model: M
): AutoCloseable {

    val tensorflowSession = TensorflowSession(Session(model.graph))

    fun evaluate(vararg inputs: DoubleArrayND): List<DoubleArrayND> =
        evaluate(inputs.toList())

    fun evaluate(
        inputs: List<DoubleArrayND>
    ): List<DoubleArrayND> =
        evaluate(
            operandList = model.outputList,
            feed = model.inputList.zip(inputs) { operand, input ->
                operand to input
            }.toMap()
        )

    fun execute(
        operation: Operation,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) {
        tensorflowSession.execute(operation, feed)
    }

    fun execute(
        target: TFOperand,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) = tensorflowSession.execute(target, feed)

    fun execute(
        targetList: List<TFOperand>,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) = tensorflowSession.execute(targetList, feed)

    fun evaluate(
        operand: Operand<*>,
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ): DoubleArrayND = tensorflowSession.evaluate(operand, targetList, feed)

    fun evaluate(
        operandList: List<TFOperand> = emptyList(),
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ): List<DoubleArrayND> = tensorflowSession.evaluate(operandList, targetList, feed)

    override fun close() {
        tensorflowSession.close()
    }

}

