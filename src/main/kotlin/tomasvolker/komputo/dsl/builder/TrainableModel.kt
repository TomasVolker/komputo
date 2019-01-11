package tomasvolker.komputo.dsl.builder

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.core.Placeholder

class TrainableModel(
    graph: Graph,
    inputList: List<Placeholder<*>>,
    outputList: List<Operand<*>>,
    val targetList: List<Placeholder<*>>,
    val loss: Operand<*>,
    val trainOperation: Operand<*>?,
    initializeOperation: Operand<*>? = null
): Model(
    graph = graph,
    inputList = inputList,
    outputList = outputList,
    initializeOperation = initializeOperation
)

open class TrainableModelBuilder(graph: Graph): ModelBuilder(graph) {

    var loss: Operand<Float> = constant(0f)

    var trainingAlgorithm: TrainingAlgorithm? = null

    override fun build(): TrainableModel {

        loss = ops.reduceMean(loss, constant(0))

        val train = trainingAlgorithm?.buildOperation(this, loss, trainableVariableList)

        return TrainableModel(
            graph = graph,
            inputList = inputList,
            outputList = outputList,
            targetList = targetList,
            loss = loss,
            trainOperation = train,
            initializeOperation = group("initialize_variables", initializationList)
        )
    }

}

inline fun trainableModel(init: TrainableModelBuilder.()->Unit): TrainableModel =
    TrainableModelBuilder(Graph()).apply(init).build()
