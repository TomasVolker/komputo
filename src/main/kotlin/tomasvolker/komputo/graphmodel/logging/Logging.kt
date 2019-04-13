package tomasvolker.komputo.graphmodel.logging

import org.tensorflow.framework.Summary
import org.tensorflow.util.Event
import tomasvolker.komputo.graphmodel.graph.computationGraph
import tomasvolker.komputo.graphmodel.graph.core.constant
import tomasvolker.komputo.graphmodel.record.tfRecordWriter
import tomasvolker.komputo.graphmodel.record.write
import java.io.File

fun tfEvent(init: Event.Builder.()->Unit): Event = Event.newBuilder().apply(init).build()
fun tfSummary(init: Summary.Builder.()->Unit): Summary = Summary.newBuilder().apply(init).build()
fun Summary.Builder.value(init: Summary.Value.Builder.()->Unit): Summary.Value =
    Summary.Value.newBuilder().apply(init).build().also { addValue(it) }

fun main() {
/*
    File("data/log/test.tfevents.${System.currentTimeMillis()}").tfRecordWriter().use {
        it.write(
            tfEvent {
                step = 2
                graphDef = computationGraph {

                    scope("escope") {

                        constant(1) + constant(2)

                    }

                }.nodeSet.toByteString()

            }
        )
    }
*/

}
