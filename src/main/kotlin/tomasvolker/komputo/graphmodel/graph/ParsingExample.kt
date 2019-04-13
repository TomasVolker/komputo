package tomasvolker.komputo.graphmodel.graph

import org.tensorflow.framework.DataType
import tomasvolker.komputo.graphmodel.graph.array.expandDims
import tomasvolker.komputo.graphmodel.graph.array.reshape
import tomasvolker.komputo.graphmodel.graph.core.constant
import tomasvolker.komputo.graphmodel.graph.core.identity
import tomasvolker.komputo.graphmodel.graph.io.*
import tomasvolker.komputo.graphmodel.proto.shape
import tomasvolker.komputo.graphmodel.proto.tensorProto
import tomasvolker.komputo.graphmodel.proto.tensorShapeProto
import tomasvolker.komputo.graphmodel.runtime.readFloats

fun main() {
/*
    File("data/small.tfrecord").tfRecordWriter().use {
        it.write(
            exampleProto {
                features {
                    feature("tensor",
                        floatArrayOf(1.2f, 5.6f, 4.8f).toTensorProto().toByteArray()
                    )
                }
            }.toByteArray()
        )
    }
*/

    val graph = computationGraph {

        val filename = constant("./data/small.tfrecord")

        val filesQueue = fifoQueue(listOf(DataType.DT_STRING), name = "filesQueue")

        queueEnqueue(filesQueue, listOf(filename), "enqueueFile")

        queueDequeue(filesQueue, "dequeueFile")

        val recordReader = tfRecordReader("reader")

        val read = readerRead(recordReader, filesQueue, "read")

        val expand = expandDims(read.outputList[1], constant(0), "expand")

        val parsed = parseExample(
            input = expand,
            names = constant(
                tensorProto {
                    dtype = DataType.DT_STRING
                    shape(0)
                    //addStringVal("tensor".toByteString())
                }
            ),
            denseKeys = listOf(
                constant("tensor")
            ),
            denseDefaults = listOf(
                constant(
                    tensorProto {
                        dtype = DataType.DT_STRING
                        shape(0)
                    }
                )
            ),
            denseTypes = listOf(DataType.DT_STRING),
            denseShapes = listOf(tensorShapeProto { }),
            name = "parseExample"
        )

        val reshape = reshape(
            input = parsed.denseValues[0],
            shape = constant(
                tensorProto {
                    dtype = DataType.DT_INT32
                    shape(0)
                }
            ),
            type = DataType.DT_STRING
        )

        val tensor = parseTensor(reshape, DataType.DT_FLOAT, "parseTensor")

        identity(tensor + tensor, "result")

    }.also { println(it) }

    graph.session {

        run {
            addTarget("enqueueFile")
        }

        val result = run {
            fetch("result")
        }

        println(result[0])

        result[0].readFloats().also {
            println(it.contentToString())
        }

    }

}
