package tomasvolker.komputo.proto

import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto
import org.tensorflow.framework.TensorShapeProto

inline fun tensorProto(block: TensorProto.Builder.()->Unit): TensorProto =
        TensorProto.newBuilder().apply(block).build()

inline fun tensorShapeProto(block: TensorShapeProto.Builder.()->Unit): TensorShapeProto =
    TensorShapeProto.newBuilder().apply(block).build()

fun tensorShapeProto(vararg sizes: Int): TensorShapeProto =
    tensorShapeProto(*sizes.map { it.toLong() }.toLongArray())

fun tensorShapeProto(vararg sizes: Long): TensorShapeProto =
    tensorShapeProto {
        sizes.forEach {
            dim { size = it }
        }
    }

inline fun TensorProto.Builder.shape(block: TensorShapeProto.Builder.()->Unit): TensorShapeProto =
    tensorShapeProto(block).also { tensorShape = it }

fun TensorProto.Builder.scalar(): TensorShapeProto = shape {  }

fun TensorProto.Builder.shape(vararg sizes: Int): TensorShapeProto =
    shape(*sizes.map { it.toLong() }.toLongArray())

fun TensorProto.Builder.shape(vararg sizes: Long): TensorShapeProto =
    shape {
        sizes.forEach {
            dim { size = it }
        }
    }

fun TensorProto.Builder.doubleValues(vararg values: Double) = values.forEach { addDoubleVal(it) }
fun TensorProto.Builder.floatValues(vararg values: Float) = values.forEach { addFloatVal(it) }
fun TensorProto.Builder.intValues(vararg values: Int) = values.forEach { addIntVal(it) }
fun TensorProto.Builder.int64Values(vararg values: Long) = values.forEach { addInt64Val(it) }
fun TensorProto.Builder.booleanValues(vararg values: Boolean) = values.forEach { addBoolVal(it) }

inline fun TensorShapeProto.Builder.dim(block: TensorShapeProto.Dim.Builder.()->Unit): TensorShapeProto.Dim =
    TensorShapeProto.Dim.newBuilder().apply(block).build().also { addDim(it) }

inline fun tensorProtoScalar(dataType: DataType, block: TensorProto.Builder.()->Unit): TensorProto =
    tensorProto {
        dtype = dataType
        scalar()
        block()
    }

fun Boolean.toTensorProto(dataType: DataType = DataType.DT_BOOL): TensorProto =
    tensorProtoScalar(dataType) { addBoolVal(this@toTensorProto) }

fun Int.toTensorProto(dataType: DataType = DataType.DT_INT32): TensorProto =
    tensorProtoScalar(dataType) { addIntVal(this@toTensorProto) }

fun Long.toTensorProto(dataType: DataType = DataType.DT_INT64): TensorProto =
    tensorProtoScalar(dataType) { addInt64Val(this@toTensorProto) }

fun Float.toTensorProto(dataType: DataType = DataType.DT_FLOAT): TensorProto =
    tensorProtoScalar(dataType) { addFloatVal(this@toTensorProto) }

fun Double.toTensorProto(dataType: DataType = DataType.DT_DOUBLE): TensorProto =
    tensorProtoScalar(dataType) { addDoubleVal(this@toTensorProto) }

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

    val tensor = tensorProto {
        dtype = DataType.DT_FLOAT
        shape(1, 2)
        floatValues(1.5f, 3.8f)
    }

    println(tensor)

}