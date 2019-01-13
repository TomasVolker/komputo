package tomasvolker.komputo.builder

import tomasvolker.komputo.performance.Timer

object VerboseTask: TrainingTask {

    private val trainingTimer = Timer()
    private val epochTimer = Timer()

    private var lastLine = ""

    private val batchLoss = mutableListOf<Double>()

    private fun renderLine(line: String) {
        clearLine()
        print(line)
        lastLine = line

    }

    private fun clearLine() {
        print("\b".repeat(lastLine.length))
        lastLine = ""
    }

    private fun nextLine() {
        println()
        lastLine = ""
    }

    override fun beforeTraining(session: TrainingContext) {

        val parameters = session.parameters

        println("""
Training on ${parameters.dataset.size} data points for ${parameters.epochs} epochs with batch size ${parameters.batchSize}
        """.trimIndent()
        )

        trainingTimer.reset()

    }

    override fun beforeEpoch(context: EpochContext) {
        println("Starting Epoch ${context.epoch}/${context.epochCount}")
        epochTimer.reset()
    }


    override fun afterBatch(context: BatchContext, loss: Double) {

        val elapsed = epochTimer.tockSeconds()
        val shownDataPoints = context.batch * context.batchSize
        val datasetSize = context.session.dataset.size
        val doneRatio = shownDataPoints / datasetSize.toDouble()

        val eta = elapsed / doneRatio - elapsed

        renderLine(
            "Trained on %d/%d metric: %g ETA: %.0f seconds".format(shownDataPoints, datasetSize, loss, eta)
        )
        batchLoss += loss
    }

    override fun afterEpoch(context: EpochContext) {
        clearLine()
        val averageLoss = batchLoss.average()
        val epochTime = epochTimer.tickSeconds()
        println("Epoch %d/%d finished on %.2f seconds, average loss: %g".format(
            context.epoch,
            context.epochCount,
            epochTime,
            averageLoss
        ))
        batchLoss.clear()
    }

    override fun afterTraining(session: TrainingContext) {
        println("Training finished in %.2f seconds".format(trainingTimer.tickSeconds()))
    }

}

fun TrainingParameters.verbose() = register(VerboseTask)
