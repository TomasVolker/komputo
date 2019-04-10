package tomasvolker.komputo.session

import tomasvolker.komputo.builder.ModelSession
import tomasvolker.komputo.builder.TrainableModel
import tomasvolker.komputo.dataset.LabeledDataset
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.doubleArray0D
import tomasvolker.numeriko.core.operations.stack

fun ModelSession<TrainableModel>.train(init: TrainingParameters.()->Unit) {
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

fun TrainingParameters.afterTraining(block: TrainingContext.()->Unit) {

    register(
        object: TrainingTask {
            override fun afterTraining(session: TrainingContext) {
                session.block()
            }
        }
    )

}

interface TrainingTask {

    fun setup(session: TrainingParameters) {}

    fun beforeTraining(session: TrainingContext) {}

    fun beforeEpoch(context: EpochContext) {}

    fun beforeBatch(context: BatchContext) {}

    fun afterBatch(context: BatchContext, loss: Double) {}

    fun afterEpoch(context: EpochContext) {}

    fun afterTraining(session: TrainingContext) {}

}

interface TrainingContext {

    val session: TrainingSession
    val dataset: LabeledDataset<DoubleArrayND, DoubleArrayND>
    val epochCount: Int
    val batchSize: Int

    val parameters get() = session.parameters
    val model get() = session.model

}

class EpochContext(
    override val session: TrainingSession,
    val epoch: Int
): TrainingContext by session

class BatchContext(
    override val session: TrainingSession,
    val batch: Int,
    val epoch: Int
): TrainingContext by session

class TrainingParameters(
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
    val modelSession: ModelSession<TrainableModel>,
    override val parameters: TrainingParameters
): TrainingContext {

    override val model: TrainableModel get() = modelSession.model

    override val session: TrainingSession
        get() = this

    private val taskList get() = parameters.taskList

    override val epochCount get() = parameters.epochs
    override val batchSize get() = parameters.batchSize

    override val dataset get() = parameters.dataset

    fun run() {

        with(modelSession) {

            taskList.forEach { it.beforeTraining(this@TrainingSession) }

            for(epoch in 1..epochCount) {

                val epochContext = EpochContext(this@TrainingSession, epoch)

                taskList.forEach { it.beforeEpoch(epochContext) }

                val batchList = dataset.shuffled().chunked(batchSize)

                for ((batchIndex, batch) in batchList.withIndex()) {

                    val batchContext = BatchContext(this@TrainingSession, batchIndex, epoch)

                    val inputTensor = batch.map { it.data }.stack(axis = 0)
                    val output = batch.map { it.label }.stack(axis = 0)
                    /*
                    val (loss, cost) = evaluate(
                        operandList = listOf(model.loss, model.cost),
                        targetList = listOf(model.optimize),
                        feed = mapOf(
                            model.inputList.first() to inputTensor,
                            model.targetList.first() to output,
                            model.trainingFactor to doubleArray0D(1.0)
                        )
                    )
                    */

                    //taskList.asReversed().forEach { it.afterBatch(batchContext, loss.as0D().get()) }
                }

                taskList.asReversed().forEach { it.afterEpoch(epochContext) }

            }

            taskList.asReversed().forEach { it.afterTraining(this@TrainingSession) }

        }

    }

}