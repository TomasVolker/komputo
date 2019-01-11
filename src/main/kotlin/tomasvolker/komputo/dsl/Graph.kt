package tomasvolker.komputo.dsl

import org.tensorflow.Graph
import org.tensorflow.op.Ops

inline fun Graph.withOps(init: Ops.()->Unit) = this.also { Ops.create(it).init() }

inline fun buildGraph(init: Ops.()->Unit) = Graph().withOps(init)

