package tomasvolker.komputo.builder

import org.tensorflow.DataType
import org.tensorflow.Operation
import tomasvolker.komputo.*
import tomasvolker.komputo.builder.optimizers.GradientDescent
import tomasvolker.komputo.builder.optimizers.Optimizer
import tomasvolker.komputo.dsl.*
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.factory.toIntArray1D

class Training(
    val targetList: List<TFPlaceholder>,
    val loss: TFOperand,
    val cost: TFOperand,
    val optimize: TFOperand,
    val filename: TFOperand,
    val save: Operation,
    val restore: TFOperand
)

class TrainableModel(
    val model: Model,
    val training: Training
): Model by model


fun trainableModel(init: TrainableModelBuilder.()->Unit) =
    TrainableModelBuilder().apply(init).build()


class TrainableModelBuilder{

    var model: Model? = null
    var loss: Metric = meanSquareError
    var optimizer: Optimizer =
        GradientDescent(1.0)

    var regularize: Boolean = true


    fun graph(block: ModelBuilder.()->Unit): Model =
        graphModel(block).also { model = it }

    fun sequential(vararg input: Int, block: SequentialBuilder.()->Unit): Model =
        sequential(input.toIntArray1D(), block)

    fun sequential(inputShape: IntArray1D, block: SequentialBuilder.()->Unit): Model =
        sequentialModel(inputShape, block).also { model = it }

    fun training(block: () -> Unit) {
        block()
    }

    fun build(): TrainableModel {

        lateinit var optimize: TFOperand

        lateinit var saveOp: Operation
        lateinit var restoreOp: TFOperand

        lateinit var filename: TFOperand

        val model = model ?: error("model is not defined")

        lateinit var targetList: List<TFPlaceholder>
        lateinit var lossOperation: TFOperand
        lateinit var costOperation: TFOperand
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


            scope("save") {
                filename = placeholder("file_name", shape = scalar, dataType = DataType.STRING)

                val tensorNames = constant(variableList.map { it.name }).asOfString()

                saveOp = ops.save(
                    filename,
                    tensorNames,
                    variableList
                )

                val restoreValues = ops.restore(
                    filename,
                    tensorNames,
                    broadcastTo(constant(""), constant(I[variableList.size])),
                    variableList.map { /*it.dataType*/ DataType.FLOAT }
                )

                val restoreOps = variableList.zip(restoreValues.outputList(0, variableList.size)) { variable, output ->
                    variable assignTo output
                }

                restoreOp = group(restoreOps)
            }


        }
        return TODO()
/*
        return TrainableModel(
            builder = model.builder,
            inputList = model.inputList,
            outputList = model.outputList,
            parameterList = model.parameterList,
            variableList = model.variableList,
            regularizationList = model.regularizationList,
            trainingFactor = model.trainingFactor,
            initializeList = model.initializeList,
            targetList = targetList,
            loss = lossOperation,
            cost = costOperation,
            optimize = optimize,
            filename = filename,
            save = saveOp,
            restore = restoreOp
        )*/
    }

}