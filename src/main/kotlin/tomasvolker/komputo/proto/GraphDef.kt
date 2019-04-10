package tomasvolker.komputo.proto

import com.google.protobuf.ByteString
import com.google.protobuf.ListValue
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.framework.*
import java.nio.FloatBuffer

inline fun graphDef(block: GraphDef.Builder.()->Unit): GraphDef =
        GraphDef.newBuilder().apply(block).build()

inline fun GraphDef.Builder.node(block: NodeDef.Builder.()->Unit): NodeDef =
        NodeDef.newBuilder().apply(block).build().also { addNode(it) }

inline fun GraphDef.Builder.versions(block: VersionDef.Builder.()->Unit): VersionDef =
    VersionDef.newBuilder().apply(block).build().also { versions = it }

fun NodeDef.Builder.input(name: String) { addInput(name) }

inline fun NodeDef.Builder.attribute(key: String, block: AttrValue.Builder.()->Unit): AttrValue =
    AttrValue.newBuilder().apply(block).build().also { putAttr(key, it) }

fun NodeDef.Builder.attribute(name: String, value: Boolean): AttrValue =
        attribute(name) { b = value }

fun NodeDef.Builder.attribute(name: String, value: Long): AttrValue =
    attribute(name) { i = value }

fun NodeDef.Builder.attribute(name: String, value: Int): AttrValue =
    attribute(name, value.toLong())

fun NodeDef.Builder.attribute(name: String, value: Float): AttrValue =
    attribute(name) { f = value }

fun NodeDef.Builder.attribute(name: String, value: Double): AttrValue =
    attribute(name, value.toFloat())

fun NodeDef.Builder.attribute(name: String, value: ByteString): AttrValue =
    attribute(name) { s = value }

fun NodeDef.Builder.attribute(name: String, value: String): AttrValue =
    attribute(name, value.toByteString())

fun NodeDef.Builder.attribute(name: String, value: ByteArray): AttrValue =
    attribute(name, value.toByteString())

fun NodeDef.Builder.attribute(name: String, value: TensorShapeProto): AttrValue =
    attribute(name) { shape = value }

fun NodeDef.Builder.attribute(name: String, value: DataType): AttrValue =
    attribute(name) { type = value }

fun NodeDef.Builder.attribute(name: String, value: TensorProto): AttrValue =
    attribute(name) { tensor = value }

fun NodeDef.Builder.attribute(name: String, value: AttrValue.ListValue): AttrValue =
    attribute(name) { list = value }

fun NodeDef.Builder.listValue(block: AttrValue.ListValue.Builder.()->Unit): AttrValue.ListValue =
    AttrValue.ListValue.newBuilder().apply(block).build()


fun main() {

    val graphDef = graphDef {

        node {
            name = "constant1"
            op = "Const"

            attribute("dtype", DataType.DT_FLOAT)
            attribute("value", floatArrayOf(1.2f, 5.0f, 2.5f, 86.1f, 89.1f).toTensorProto())

        }

        node {
            name = "constant2"
            op = "Const"

            attribute("dtype", DataType.DT_INT32)
            attribute("value", 2.toTensorProto())

        }

        node {
            name = "gather"
            op = "Gather"

            attribute("Tparams", DataType.DT_FLOAT)
            attribute("Tindices", DataType.DT_INT32)

            input("constant1")
            input("constant2")

        }

    }.also { println(it) }

    val graph = Graph().apply {
        importGraphDef(graphDef.toByteArray())
        /*addGradients(
            operation("gather").output<Any>(0),
            arrayOf(operation("constant1").output<Any>(0))
        )*/
    }

    Session(graph).use { session ->

        val result = session.runner()
            .fetch("gather")
            .run()

        val buffer = FloatBuffer.allocate(1)

        result[0].writeTo(buffer)

        buffer.rewind()

        println(buffer[0])

    }

    graph.close()

}