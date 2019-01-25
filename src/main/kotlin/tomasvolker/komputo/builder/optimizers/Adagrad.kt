package tomasvolker.komputo.builder.optimizers

import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.builder.ModelBuilder
import tomasvolker.komputo.dsl.localName
import tomasvolker.komputo.dsl.shape

class Adagrad(
    val rate: Double = 0.01,
    val epsilon: Double = 1e-8
): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)

        return parameterList.mapIndexed { i, parameter ->

            val parameterShape = parameter.shape

            val accumulator = variable(
                "${parameter.localName}_accumulator",
                shape = parameterShape,
                dataType = defaultFloatDataType,
                initialValue = broadcastTo(constant(epsilon), constant(parameterShape))
            )

            ops.applyAdagrad(
                parameter.asOfAny(),
                accumulator.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i)
            )
        }
    }


}
