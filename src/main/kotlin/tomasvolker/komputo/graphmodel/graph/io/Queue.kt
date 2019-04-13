package tomasvolker.komputo.graphmodel.graph.io

import org.tensorflow.framework.DataType
import tomasvolker.komputo.graphmodel.graph.GraphNode

interface Queue: GraphNode {

    val componentTypes: List<DataType>

}