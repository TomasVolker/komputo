package tomasvolker.komputo.builder

import org.tensorflow.op.core.PlaceholderWithDefault
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.TFVariable

interface Model {

    val builder: ModelBuilder

    val initializeList: List<TFOperand>
    val inputList: List<TFOperand>
    val parameterList: List<TFVariable>
    val variableList: List<TFVariable>
    val outputList: List<TFOperand>

    val regularizationList: List<TFOperand>
    val trainingFactor: PlaceholderWithDefault<*>


    val inputSize: Int get() = inputList.size
    val outputSize: Int get() = outputList.size

    val graph get() = builder.graph

}

class TensorflowModel(
    override val builder: ModelBuilder,
    override val inputList: List<TFOperand>,
    override val outputList: List<TFOperand>,
    override val parameterList: List<TFVariable>,
    override val variableList: List<TFVariable>,
    override val regularizationList: List<TFOperand> = mutableListOf(),
    override val trainingFactor: PlaceholderWithDefault<*>,
    override val initializeList: List<TFOperand> = emptyList()
): Model

