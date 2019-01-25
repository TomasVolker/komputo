package tomasvolker.komputo.builder.optimizers


import org.tensorflow.op.core.Variable
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.builder.ModelBuilder

interface Optimizer {

    fun buildOperations(
        builder: ModelBuilder,
        loss: TFOperand,
        parameterList: List<Variable<*>>
    ): TFOperand

}

