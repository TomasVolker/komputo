package tomasvolker.komputo.graphmodel.proto

import tomasvolker.komputo.graphmodel.utils.toByteString
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto
import org.tensorflow.framework.TensorShapeProto
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun tensorProto(block: TensorProto.Builder.()->Unit): TensorProto =
        TensorProto.newBuilder().apply(block).build()

fun TensorProto.Builder.shape(block: TensorShapeProto.Builder.()->Unit): TensorShapeProto =
        tensorShapeProto(block).also { tensorShape = it }

fun TensorProto.Builder.shape(vararg shape: Long): TensorShapeProto =
        shape { shape.forEach { dim(it) } }

fun TensorProto.Builder.shape(vararg shape: Int): TensorShapeProto =
        shape { shape.forEach { dim(it) } }

fun TensorProto.Builder.scalar(): TensorShapeProto = shape {}

fun tensorShapeProto(vararg sizes: Int): TensorShapeProto =
    tensorShapeProto(*sizes.map { it.toLong() }.toLongArray())

fun tensorShapeProto(vararg sizes: Long): TensorShapeProto =
    tensorShapeProto {
        sizes.forEach {
            dim { size = it }
        }
    }

fun tensorShapeProto(block: TensorShapeProto.Builder.()->Unit): TensorShapeProto =
        TensorShapeProto.newBuilder().apply(block).build()

fun TensorShapeProto.Builder.dim(block: TensorShapeProto.Dim.Builder.()->Unit): TensorShapeProto.Dim =
        tensorShapeDim(block).also { addDim(it) }

fun TensorShapeProto.Builder.dim(size: Int): TensorShapeProto.Dim = dim(size.toLong())
fun TensorShapeProto.Builder.dim(size: Long): TensorShapeProto.Dim = dim { this.size = size }

fun tensorShapeDim(block: TensorShapeProto.Dim.Builder.()->Unit): TensorShapeProto.Dim =
        TensorShapeProto.Dim.newBuilder().apply(block).build()

fun tensorProtoScalar(type: DataType, block: TensorProto.Builder.()->Unit): TensorProto =
        tensorProto {
            dtype = type
            scalar()
            block()
        }

fun tensorProtoScalar(value: Int, type: DataType = DataType.DT_INT32): TensorProto =
        tensorProtoScalar(type) { addIntVal(value) }

fun tensorProtoScalar(value: Long, type: DataType = DataType.DT_INT64): TensorProto =
        tensorProtoScalar(type) { addInt64Val(value) }

fun tensorProtoScalar(value: Boolean, type: DataType = DataType.DT_BOOL): TensorProto =
        tensorProtoScalar(type) { addBoolVal(value) }

fun tensorProtoScalar(value: Float, type: DataType = DataType.DT_FLOAT): TensorProto =
        tensorProtoScalar(type) { addFloatVal(value) }

fun tensorProtoScalar(value: Double, type: DataType = DataType.DT_DOUBLE): TensorProto =
        tensorProtoScalar(type) { addDoubleVal(value) }

fun tensorProtoScalar(value: String): TensorProto =
        tensorProtoScalar(DataType.DT_STRING) { addStringVal(value.toByteString()) }

fun TensorProto.Builder.doubleValues(vararg values: Double) = values.forEach { addDoubleVal(it) }
fun TensorProto.Builder.floatValues(vararg values: Float) = values.forEach { addFloatVal(it) }
fun TensorProto.Builder.intValues(vararg values: Int) = values.forEach { addIntVal(it) }
fun TensorProto.Builder.int64Values(vararg values: Long) = values.forEach { addInt64Val(it) }
fun TensorProto.Builder.booleanValues(vararg values: Boolean) = values.forEach { addBoolVal(it) }

fun IntArray.toTensorShapeProto(): TensorShapeProto = tensorShapeProto(*this)
fun LongArray.toTensorShapeProto(): TensorShapeProto = tensorShapeProto(*this)

fun BooleanArray.toTensorProto(dataType: DataType = DataType.DT_BOOL): TensorProto =
    tensorProto {
        dtype = dataType
        shape(this@toTensorProto.size)
        this@toTensorProto.forEach { addBoolVal(it) }
    }

fun IntArray.toTensorProto(dataType: DataType = DataType.DT_INT32): TensorProto =
    tensorProto {
        dtype = dataType
        shape(this@toTensorProto.size)
        this@toTensorProto.forEach { addIntVal(it) }
    }

fun LongArray.toTensorProto(dataType: DataType = DataType.DT_INT64): TensorProto =
    tensorProto {
        dtype = dataType
        shape(this@toTensorProto.size)
        this@toTensorProto.forEach { addInt64Val(it) }
    }

fun FloatArray.toTensorProto(dataType: DataType = DataType.DT_FLOAT): TensorProto =
    tensorProto {
        dtype = dataType
        shape(this@toTensorProto.size)
        this@toTensorProto.forEach { addFloatVal(it) }
    }

fun DoubleArray.toTensorProto(dataType: DataType = DataType.DT_DOUBLE): TensorProto =
    tensorProto {
        dtype = dataType
        shape(this@toTensorProto.size)
        this@toTensorProto.forEach { addDoubleVal(it) }
    }

fun main() {

    val array = floatArrayOf(
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
    )

    val byteBuffer = ByteBuffer.allocate(4 * 6).order(ByteOrder.LITTLE_ENDIAN)
    array.forEach { byteBuffer.putFloat(it) }
    byteBuffer.rewind()

    val tensor = tensorProto {
        dtype = DataType.DT_FLOAT
        shape(2, 3)
        tensorContent = byteBuffer.toByteString()
    }

    File("data/small.value").writeBytes(tensor.toByteArray())

}