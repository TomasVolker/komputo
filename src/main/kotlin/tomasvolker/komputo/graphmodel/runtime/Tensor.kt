package tomasvolker.komputo.graphmodel.runtime

import org.tensorflow.Tensor
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto
import java.nio.*

fun Tensor<*>.readBytes(): ByteArray = use {
    val buffer = ByteBuffer.allocate(numBytes()).order(ByteOrder.LITTLE_ENDIAN)
    it.writeTo(buffer)
    buffer.array()
}

fun Tensor<*>.readLongs(): LongArray = use {
    val buffer = LongBuffer.allocate(numElements())
    it.writeTo(buffer)
    buffer.array()
}

fun Tensor<*>.readInts(): IntArray = use {
    val buffer = IntBuffer.allocate(numElements())
    it.writeTo(buffer)
    buffer.array()
}

fun Tensor<*>.readDoubles(): DoubleArray = use {
    val buffer = DoubleBuffer.allocate(numElements())
    it.writeTo(buffer)
    buffer.array()

}

fun Tensor<*>.readFloats(): FloatArray = use {
    val buffer = FloatBuffer.allocate(numElements())
    it.writeTo(buffer)
    buffer.array()
}

/*
fun TensorProto.toTensor(): Tensor<*> =
        when(dtype) {
            DataType.DT_FLOAT ->
                Tensor.create()
        }

 */
