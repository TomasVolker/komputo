package tomasvolker.komputo.builder.optimizers

import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.builder.ModelBuilder
import tomasvolker.komputo.builder.group

abstract class GradientAlgorithm: Optimizer {

    override fun buildOperations(
        builder: ModelBuilder,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): TFOperand {

        with(builder) {

            var result: List<TFOperand>? = null

            scope("gradient") {

                val grad = gradients(loss, parameterList)

                result = buildOptimzeOperations(builder, parameterList, grad)

            }

            scope("optmizer") {
                return group("optimize", result ?: error(""))
            }

        }

    }

    fun buildOptimzeOperations(
        builder: ModelBuilder,
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> = builder.buildOperations(parameterList, gradients)

    abstract fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand>

}