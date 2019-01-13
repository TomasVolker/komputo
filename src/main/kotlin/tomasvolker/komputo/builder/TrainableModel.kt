package tomasvolker.komputo.builder

import org.tensorflow.op.Ops
import org.tensorflow.op.core.PlaceholderWithDefault
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.TFPlaceholder
import tomasvolker.komputo.TFVariable
import tomasvolker.komputo.dsl.dataType
import tomasvolker.komputo.dsl.localName
import tomasvolker.komputo.dsl.shape
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D

class TrainableModel(
    builder: ModelBuilder,
    inputList: List<TFOperand>,
    outputList: List<TFOperand>,
    parameterList: List<TFVariable>,
    regularizationList: List<TFOperand>,
    trainingFactor: PlaceholderWithDefault<*>,
    initializeList: List<TFOperand>,
    val targetList: List<TFPlaceholder>,
    val loss: TFOperand,
    val cost: TFOperand,
    val optimize: TFOperand
): Model(
    builder = builder,
    inputList = inputList,
    outputList = outputList,
    parameterList = parameterList,
    regularizationList = regularizationList,
    trainingFactor = trainingFactor,
    initializeList = initializeList
)


fun trainableModel(init: TrainableModelBuilder.()->Unit) =
    TrainableModelBuilder().apply(init).build()



class TrainableModelBuilder{

    var model: Model? = null
    var loss: LossFunction = meanSquareError
    var optimizer: Optimizer = GradientDescent(1.0)

    var regularize: Boolean = true


    fun graph(block: ModelBuilder.()->Unit): Model =
        graphModel(block).also { model = it }

    fun sequential(vararg input: Int, block: SequentialBuilder.()->Unit): Model =
        sequential(input.toIntArray1D(), block)

    fun sequential(inputShape: IntArray1D, block: SequentialBuilder.()->Unit): Model =
        sequentialModel(inputShape, block).also { model = it }

    fun training(block: ModelBuilder.() -> Unit) {
        model?.builder?.block()
    }

    fun build(): TrainableModel {

        val model = model ?: error("model is not defined")

        lateinit var targetList: List<TFPlaceholder>
        lateinit var lossOperation: TFOperand
        lateinit var costOperation: TFOperand
        lateinit var optimize: TFOperand

        with(model.builder) {

            scope("target") {
                targetList = model.outputList.map { output ->
                    placeholder(
                        name = "${output.localName}_target",
                        dataType = output.dataType,
                        shape = output.shape
                    )
                }
            }

            scope("optimizer") {

                lossOperation = loss.buildOperations(
                    this,
                    model.outputList.first(),
                    targetList.first()
                )

                if (regularizationList.isEmpty() || !regularize)
                    costOperation = lossOperation
                else
                    costOperation = lossOperation + regularizationList.reduce { acc, op -> acc + op }

                optimize = optimizer.buildOperations(
                    this,
                    costOperation,
                    model.parameterList
                )
            }


        }

        return TrainableModel(
            builder = model.builder,
            inputList = model.inputList,
            outputList = model.outputList,
            parameterList = model.parameterList,
            regularizationList = model.regularizationList,
            trainingFactor = model.trainingFactor,
            initializeList = model.initializeList,
            targetList = targetList,
            loss = lossOperation,
            cost = costOperation,
            optimize = optimize
        )
    }

}