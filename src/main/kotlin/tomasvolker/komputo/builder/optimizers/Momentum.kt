package tomasvolker.komputo.builder.optimizers

import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.builder.ModelBuilder
import tomasvolker.komputo.dsl.localName
import tomasvolker.komputo.dsl.shape

class Momentum(
    val rate: Double,
    val momentum: Double = 0.9
): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)
        val momentum = constant(momentum)

        return parameterList.mapIndexed { i, parameter ->

            val parameterShape = parameter.shape

            val accumulator = variable(
                "${parameter.localName}_accumulator",
                shape = parameterShape,
                dataType = defaultFloatDataType,
                initialValue = broadcastTo(constant(0.0), constant(parameterShape))
            )

            ops.applyMomentum(
                parameter.asOfAny(),
                accumulator.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i),
                momentum.asOfAny()
            )
        }
    }

}