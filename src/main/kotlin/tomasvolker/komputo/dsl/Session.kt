package tomasvolker.komputo.dsl

import org.tensorflow.*
import org.tensorflow.op.core.SaveV2
import tomasvolker.komputo.TFOperand
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.arraynd.generic.ArrayND


open class TensorflowSession(
    val session: Session
): AutoCloseable {

    inner class Runner {

        private val feed = mutableMapOf<Any, ArrayND<*>>()
        private val targetList = mutableListOf<Any>()
        private val fetchList = mutableListOf<Any>()

        fun target(operand: TFOperand) {
            targetList += operand.asOutput()
        }

        fun target(operation: String) {
            targetList += operation
        }

        fun target(operation: Operation) {
            targetList += operation
        }

        fun fetch(operand: TFOperand) {
            fetchList += operand.asOutput()
        }

        fun fetch(operand: String) {
            fetchList += operand
        }

        fun feed(map: Map<TFOperand, ArrayND<*>>) {
            map.forEach { feed(it.key, it.value) }
        }

        @JvmName("feedNames")
        fun feed(map: Map<String, ArrayND<*>>) {
            map.forEach { feed(it.key, it.value) }
        }

        fun feed(vararg pairs: Pair<TFOperand, ArrayND<*>>) {
            pairs.forEach { feed(it.first, it.second) }
        }

        @JvmName("feedNames")
        fun feed(vararg pairs: Pair<String, ArrayND<*>>) {
            pairs.forEach { feed(it.first, it.second) }
        }

        fun feed(operand: TFOperand, array: ArrayND<*>) {
            feed[operand] = array
        }

        fun feed(operand: String, array: ArrayND<*>) {
            feed[operand] = array
        }

        fun run(): List<DoubleArrayND> {

            val feedTensors = feed.mapValues { it.value.toTensor() }

            try {

                val resultList = session.runner().apply {

                    feedTensors.forEach { operand, tensor ->
                        when(operand) {
                            is String -> feed(operand, tensor)
                            is TFOperand -> feed(operand.asOutput(), tensor)
                        }
                    }

                    targetList.forEach {
                        when(it) {
                            is String -> addTarget(it)
                            is TFOperand -> addTarget(it)
                            is Operation -> addTarget(it)
                        }
                    }

                    fetchList.forEach {
                        when(it) {
                            is String -> fetch(it)
                            is TFOperand -> fetch(it)
                        }
                    }

                }.run()

                try {
                    return resultList.map { it.toDoubleNDArray() }
                } finally {
                    resultList.forEach { it.close() }
                }

            } finally {
                feedTensors.forEach { _, tensor -> tensor.close() }
            }
        }

    }

    fun execute(init: Runner.()->Unit): List<DoubleArrayND> =
        Runner().apply(init).run()

    fun execute(
        operation: Operation,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) {
        evaluate(operationList = listOf(operation), feed = feed)
    }

    fun execute(
        target: TFOperand,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) {
        execute(listOf(target), feed)
    }

    fun execute(
        targetList: List<TFOperand>,
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ) {
        evaluate(
            targetList = targetList,
            feed = feed
        )
    }

    fun evaluate(
        operand: Operand<*>,
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, ArrayND<*>> = emptyMap()
    ): DoubleArrayND = evaluate(listOf(operand), targetList, feed).first()

    fun evaluate(
        operandList: List<TFOperand> = emptyList(),
        targetList: List<TFOperand> = emptyList(),
        feed: Map<TFOperand, ArrayND<*>> = emptyMap(),
        operationList: List<Operation> = emptyList()
    ): List<DoubleArrayND> = execute {
        feed.forEach { operand, tensor ->
            feed(operand, tensor)
        }
        targetList.forEach {
            target(it)
        }
        operationList.forEach {
            target(it)
        }
        operandList.forEach { operand ->
            fetch(operand)
        }
    }

    fun evaluate(
        fetchList: List<String> = emptyList(),
        targetList: List<String> = emptyList(),
        feed: Map<String, ArrayND<*>> = emptyMap()
    ): List<DoubleArrayND> = execute {
        feed.forEach { operand, tensor ->
            feed(operand, tensor)
        }
        targetList.forEach {
            target(it)
        }
        fetchList.forEach { operand ->
            fetch(operand)
        }
    }

    override fun close() {
        session.close()
    }

}


inline fun <T> session(graph: Graph, block: TensorflowSession.()->T) = TensorflowSession(Session(graph)).use(block)


