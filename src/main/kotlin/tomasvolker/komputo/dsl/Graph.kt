package tomasvolker.komputo.dsl

import org.tensorflow.Graph
import org.tensorflow.OperationBuilder
import org.tensorflow.op.Ops

inline fun Graph.withOps(init: Ops.()->Unit) = this.also { Ops.create(it).init() }

inline fun buildGraph(init: Ops.()->Unit) = Graph().withOps(init)


inline fun Graph.buildOp(
    operation: String,
    name: String,
    init: OperationBuilder.()->Unit = {}
) =
    opBuilder(operation, name).apply(init).build()
