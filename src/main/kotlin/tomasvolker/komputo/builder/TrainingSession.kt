package tomasvolker.komputo.builder

import org.tensorflow.op.Ops
import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.TFPlaceholder
import tomasvolker.komputo.dataset.LabeledDataset
import tomasvolker.komputo.dsl.dataType
import tomasvolker.komputo.dsl.localName
import tomasvolker.komputo.dsl.placeholder
import tomasvolker.komputo.dsl.shape
import tomasvolker.komputo.performance.stack
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.doubleArray0D

fun ModelSession.train(init: TrainingParameters.()->Unit) {
    TrainingSession(this, TrainingParameters().apply(init)).run()
}

fun TrainingParameters.afterEpoch(block: EpochContext.()->Unit) {

    register(
        object: TrainingTask {
            override fun afterEpoch(context: EpochContext) {
                context.block()
            }
        }
    )

}

fun TrainingParameters.afterTraining(block: TrainingSession.()->Unit) {

    register(
        object: TrainingTask {
            override fun afterTraining(session: TrainingSession) {
                session.block()
            }
        }
    )

}

interface TrainingTask {

    fun setup(session: TrainingParameters) {}

    fun beforeTraining(session: TrainingSession) {}

    fun beforeEpoch(context: EpochContext) {}

    fun beforeBatch(context: BatchContext) {}

    fun afterBatch(context: BatchContext, loss: Double) {}

    fun afterEpoch(context: EpochContext) {}

    fun afterTraining(session: TrainingSession) {}

}

interface TrainingContext {

    val dataset: LabeledDataset<DoubleArrayND, DoubleArrayND>
    val epochCount: Int
    val batchSize: Int

}

class EpochContext(
    val session: TrainingSession,
    val epoch: Int
): TrainingContext by session

class BatchContext(
    val session: TrainingSession,
    val batch: Int,
    val epoch: Int
): TrainingContext by session

class TrainingParameters(
    var loss: (Ops, TFOperand, TFOperand)->TFOperand = ::meanSquareError,
    var optimizer: Optimizer = GradientDescent(1.0),
    var dataset: LabeledDataset<DoubleArrayND, DoubleArrayND> = emptyList(),
    var epochs: Int = 1,
    var batchSize: Int = 32,
    var regularize: Boolean = true,
    var dropout: Boolean = true
) {

    val taskList: MutableList<TrainingTask> = mutableListOf()

    fun register(task: TrainingTask) {
        taskList += task
        task.setup(this)
    }

}



class TrainingSession(
    val session: ModelSession,
    val parameters: TrainingParameters
): TrainingContext {

    private val taskList get() = parameters.taskList

    private lateinit var lossOperation: TFOperand
    private lateinit var costOperation: TFOperand

    private lateinit var targetList: List<TFPlaceholder>

    override val epochCount get() = parameters.epochs
    override val batchSize get() = parameters.batchSize

    override val dataset get() = parameters.dataset

    val model: Model get() = session.model

    private fun buildTrainingOperations(): OptimizerOperations {

        val ops = Ops.create(model.graph)

        with(ops.withName("target")) {

            targetList = model.outputList.map { output ->
                placeholder(
                    name = "${output.localName}_target",
                    dtype = output.dataType,
                    shape = output.shape
                )
            }

        }

        lossOperation = parameters.loss(
            ops.withName("optimizer"),
            model.outputList.first(),
            targetList.first()
        )

        costOperation = lossOperation

        return parameters.optimizer.buildOperations(
            ops.withName("optimizer"),
            costOperation,
            model.parameterList
        )

    }

    fun run() {

        val (initOptimizer, optimize) = buildTrainingOperations()

        with(session) {

            model.initializeOperation?.let {
                execute(it)
            }

            execute(initOptimizer)

            taskList.forEach { it.beforeTraining(this@TrainingSession) }

            for(epoch in 1..epochCount) {

                val epochContext = EpochContext(this@TrainingSession, epoch)

                taskList.forEach { it.beforeEpoch(epochContext) }

                val batchList = dataset.shuffled().chunked(batchSize)

                for ((batchIndex, batch) in batchList.withIndex()) {

                    val batchContext = BatchContext(this@TrainingSession, batchIndex, epoch)

                    val inputTensor = stack(batch.map { it.data })
                    val output = stack(batch.map { it.label })

                    val (loss, cost) = evaluate(
                        operandList = listOf(lossOperation, costOperation),
                        targetList = listOf(optimize),
                        feed = mapOf(
                            model.inputList.first() to inputTensor,
                            targetList.first() to output,
                            model.trainingFactor to doubleArray0D(1.0)
                        )
                    )

                    taskList.asReversed().forEach { it.afterBatch(batchContext, loss.as0D().get()) }
                }

                taskList.asReversed().forEach { it.afterEpoch(epochContext) }

            }

            taskList.asReversed().forEach { it.afterTraining(this@TrainingSession) }

        }

    }

}