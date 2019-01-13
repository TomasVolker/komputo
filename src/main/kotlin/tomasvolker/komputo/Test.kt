package tomasvolker.komputo

import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.op.Operands
import org.tensorflow.op.Ops
import tomasvolker.komputo.dsl.name
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.arraynd.double.unsafeGetView
import tomasvolker.numeriko.core.interfaces.factory.doubleArray2D
import tomasvolker.numeriko.core.interfaces.factory.nextDoubleArray2D
import tomasvolker.komputo.dsl.toDoubleNDArray
import tomasvolker.komputo.dsl.toTensor
import tomasvolker.numeriko.core.interfaces.factory.doubleArray0D
import kotlin.random.Random

fun main() {

    Graph().use { graph ->

        val ops = Ops.create(graph)

        val var1 = ops.variable(Shape.scalar(), java.lang.Float::class.java)
        val set = ops.assign(var1.asOfNumber(), ops.constant(8.0f).asOfNumber())

        val filename = ops.constant("graph.pb".toByteArray(Charsets.US_ASCII))
        val varNameTensor = ops.constant(arrayOf(var1.name.toByteArray(Charsets.US_ASCII)))
        val varName = ops.constant(var1.name.toByteArray(Charsets.US_ASCII))

        val save = graph.opBuilder("Save", "Save").apply {
            addInput(filename.asOutput())
            addInput(varNameTensor.asOutput())
            addInputList(listOf(var1).map { it.asOutput() }.toTypedArray())
        }.build()

        val restore = graph.opBuilder("Restore", "Restore").apply {
            addInput(filename.asOutput())
            addInput(varName.asOutput())
            setAttr("dt", DataType.FLOAT)
        }.build()

        Session(graph).use { session ->
/*
            session.runner()
                .addTarget(set)
                .run()

            session.runner()
                .addTarget(save)
                .run()
*/

            val value = session.runner()
                .fetch(restore.output<Float>(0))
                .run()

            println("value: ${value.map { it.toDoubleNDArray() }}")

            val result = session.runner()
                .fetch(var1)
                .run().first()

            println(result.toDoubleNDArray())

            result.close()

        }

    }

}
