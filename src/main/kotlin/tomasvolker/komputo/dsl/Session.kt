package tomasvolker.komputo.dsl

import org.tensorflow.*
import tomasvolker.komputo.TFOperand
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND


open class TensorflowSession(
    val session: Session
): AutoCloseable {

    fun execute(
        target: TFOperand,
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ) {
        execute(listOf(target), feed)
    }

    fun execute(
        targetList: List<TFOperand>,
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ) {
        evaluate(
            targetList = targetList,
            feed = feed
        )
    }

    fun evaluate(
        operand: Operand<*>,
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ): DoubleArrayND = evaluate(listOf(operand), targetList, feed).first()

    fun evaluate(
        operandList: List<TFOperand> = emptyList(),
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, DoubleArrayND> = emptyMap()
    ): List<DoubleArrayND> {

        val tensorMap = feed.mapValues { it.value.toTensor() }

        try {

            val resultList = session.runner().apply {
                tensorMap.forEach { operand, tensor ->
                    feed(operand, tensor)
                }
                targetList.forEach {
                    addTarget(it)
                }
                operandList.forEach { operand ->
                    fetch(operand)
                }
            }.run()

            try {

                return resultList.map { it.toDoubleNDArray() }

            } finally {
                resultList.forEach { it.close() }
            }

        } finally {
            tensorMap.forEach { _, tensor -> tensor.close() }
        }
    }

    override fun close() {
        session.close()
    }

}

fun Session.Runner.feed(operand: TFOperand, input: Tensor<*>): Session.Runner =
    this.also { feed(operand.asOutput(), input) }


inline fun <T> session(graph: Graph, block: TensorflowSession.()->T) = TensorflowSession(Session(graph)).use(block)
