package tomasvolker.komputo.graphmodel.graph

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor

fun ComputationGraph.toGraph(): Graph = Graph().apply {
    importGraphDef(this@toGraph.toGraphDef().toByteArray())
}
inline fun <T> ComputationGraph.useGraph(block: (graph: Graph)->T): T = toGraph().use(block)

fun ComputationGraph.startSession(): Session = Session(toGraph())
fun <T> ComputationGraph.session(block: Session.()->T): T = startSession().use(block)

inline fun Session.run(init: Session.Runner.()->Unit): List<Tensor<*>> = runner().apply(init).run()
