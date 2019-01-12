package tomasvolker.komputo.builder

import org.tensorflow.Operand
import org.tensorflow.Session
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND

fun <M: Model> session(model: M, block: ModelSession<M>.()->Unit) {
    ModelSession(model).use { session ->

        model.initializeList.forEach {
            session.execute(it)
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
        target: TFOperand,
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ) = tensorflowSession.execute(target, feed)

    fun execute(
        targetList: List<TFOperand>,
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ) = tensorflowSession.execute(targetList, feed)

    fun evaluate(
        operand: Operand<*>,
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ): DoubleArrayND = tensorflowSession.evaluate(operand, targetList, feed)

    fun evaluate(
        operandList: List<TFOperand> = emptyList(),
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ): List<DoubleArrayND> = tensorflowSession.evaluate(operandList, targetList, feed)

    override fun close() {
        tensorflowSession.close()
    }

}

