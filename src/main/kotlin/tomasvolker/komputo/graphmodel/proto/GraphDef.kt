package tomasvolker.komputo.graphmodel.proto

import tomasvolker.komputo.graphmodel.graph.OperandRef
import tomasvolker.komputo.graphmodel.utils.toByteString
import org.tensorflow.framework.*

inline fun graphDef(init: GraphDef.Builder.()->Unit): GraphDef = GraphDef.newBuilder().apply(init).build()
inline fun nodeDef(init: NodeDef.Builder.()->Unit): NodeDef = NodeDef.newBuilder().apply(init).build()
inline fun nodeDef(operation: String, init: NodeDef.Builder.()->Unit): NodeDef =
        nodeDef {
            op = operation
            init()
        }
inline fun attrValue(init: AttrValue.Builder.()->Unit): AttrValue = AttrValue.newBuilder().apply(init).build()
inline fun attrListValue(init: AttrValue.ListValue.Builder.()->Unit): AttrValue.ListValue =
        AttrValue.ListValue.newBuilder().apply(init).build()

inline fun GraphDef.Builder.node(
        operation: String = "",
        init: NodeDef.Builder.()->Unit
): NodeDef = nodeDef {
    op = operation
    init()
}.also { addNode(it) }

fun NodeDef.Builder.input(input: String) { addInput(input) }
fun NodeDef.Builder.input(input: NodeDef) { addInput(input.name) }
fun NodeDef.Builder.input(input: OperandRef) { addInput(input.name) }

inline fun NodeDef.Builder.attr(key: String, init: AttrValue.Builder.()->Unit): AttrValue =
        attrValue(init).also { putAttr(key, it) }

fun NodeDef.Builder.attr(key: String, value: Boolean): AttrValue = attr(key) { b = value }
fun NodeDef.Builder.attr(key: String, value: Long): AttrValue = attr(key) { i = value }
fun NodeDef.Builder.attr(key: String, value: Int): AttrValue = attr(key, value.toLong())
fun NodeDef.Builder.attr(key: String, value: Float): AttrValue = attr(key) { f = value }
fun NodeDef.Builder.attr(key: String, value: Double): AttrValue = attr(key, value.toFloat())
fun NodeDef.Builder.attr(key: String, value: TensorProto): AttrValue = attr(key) { tensor = value }
fun NodeDef.Builder.attr(key: String, value: TensorShapeProto): AttrValue = attr(key) { shape = value }
fun NodeDef.Builder.attr(key: String, value: DataType): AttrValue = attr(key) { type = value }
fun NodeDef.Builder.attr(key: String, value: String): AttrValue = attr(key) { s = value.toByteString() }

val NodeDef.type: DataType get() =
    getAttrOrDefault("T", null)?.type ?:
    getAttrOrDefault("dtype", null)?.type ?:
    getAttrOrDefault("type", null)?.type ?:
    getAttrOrDefault("out_type", null)?.type ?:
    getAttrOrDefault("DstT", null)?.type ?:
    throw NoSuchElementException("Node $name contains no type attribute")

fun NodeDef.attr(key: String): AttrValue = getAttrOrThrow(key)
fun NodeDef.attr(key: String, default: AttrValue?): AttrValue? = getAttrOrDefault(key, default)
