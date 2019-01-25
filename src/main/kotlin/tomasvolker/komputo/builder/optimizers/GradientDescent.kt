package tomasvolker.komputo.builder.optimizers

import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.asOfAny
import tomasvolker.komputo.builder.ModelBuilder

class GradientDescent(val rate: Double): GradientAlgorithm() {

    override fun ModelBuilder.buildOperations(
        parameterList: List<Variable<*>>,
        gradients: Gradients
    ): List<TFOperand> {
        val rate = constant(rate)

        return parameterList.mapIndexed { i, parameter ->
            ops.applyGradientDescent(
                parameter.asOfAny(),
                rate.asOfAny(),
                gradients.dy(i)
            )
        }
    }

}