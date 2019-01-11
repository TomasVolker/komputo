package tomasvolker.komputo

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.op.Ops
import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.arraynd.double.unsafeGetView
import tomasvolker.numeriko.core.interfaces.factory.doubleArray2D
import tomasvolker.numeriko.core.interfaces.factory.nextDoubleArray2D
import tomasvolker.komputo.dsl.toDoubleNDArray
import tomasvolker.komputo.dsl.toTensor
import kotlin.random.Random

fun main() {

    val image = Random.nextDoubleArray2D(8, 8)
    val filter = doubleArray2D(1, 1) { _, _ -> 1.0 }

    val expected = image.filter2D(filter)

    println(image)
    println(expected)

    Graph().use { graph ->

        val ops = Ops.create(graph)

        val p1 = ops.placeholder(java.lang.Float::class.java)
        val p2 = ops.placeholder(java.lang.Float::class.java)

        val rehsaped1 = ops.reshape(p1, ops.constant(intArrayOf(1, 8, 8, 1)))
        val rehsaped2 = ops.reshape(p2, ops.constant(intArrayOf(1, 1, 1, 1)))

        val op = ops.conv2D(
            rehsaped1, rehsaped2,
            I[1, 1, 1, 1].toList().map { it.toLong() },
            "SAME"
        )


        Session(graph).use { session ->

            val tensor1 = image.toTensor()
            val tensor2 = filter.toTensor()

            val result = session.runner()
                .feed(p1.output(), tensor1)
                .feed(p2.output(), tensor2)
                .fetch(op)
                .run().first()

            println("result: \n${result.toDoubleNDArray().unsafeGetView(0, All, All, 0)}")

            result.close()
            tensor1.close()
            tensor2.close()

        }

    }

}
