package tomasvolker.komputo

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.types.UInt8
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.intArray1D

fun DataType.toClass(): Class<*> = when(this) {
    DataType.FLOAT -> java.lang.Float::class.java
    DataType.DOUBLE -> java.lang.Double::class.java
    DataType.INT32 -> java.lang.Integer::class.java
    DataType.UINT8 -> UInt8::class.java
    DataType.INT64 -> java.lang.Long::class.java
    DataType.BOOL -> java.lang.Boolean::class.java
    DataType.STRING -> java.lang.String::class.java
}


fun Double.castTo(dataType: DataType): Any = when(dataType) {
    DataType.FLOAT -> toFloat()
    DataType.DOUBLE -> this
    DataType.INT32 -> error("Cannot cast Double to INT32")
    DataType.UINT8 -> error("Cannot cast Double to UINT8")
    DataType.INT64 -> error("Cannot cast Double to INT64")
    DataType.BOOL -> error("Cannot cast Double to BOOL")
    DataType.STRING -> error("Cannot cast Double to STRING")
}

fun Int.castTo(dataType: DataType): Any = when(dataType) {
    DataType.FLOAT -> toFloat()
    DataType.DOUBLE -> toDouble()
    DataType.INT32 -> this
    DataType.UINT8 -> toByte()
    DataType.INT64 -> toLong()
    DataType.BOOL -> error("Cannot cast Int to BOOL")
    DataType.STRING -> error("Cannot cast Int to STRING")
}


fun Shape.toIntArray1D() = intArray1D(numDimensions()) { i -> size(i).toInt() }

fun IntArray1D?.toShape(): Shape =
    if (this == null)
        Shape.unknown()
    else
        Shape.make(
            this[0].toLong(),
            *this.drop(1).map { it.toLong() }.toLongArray()
        )

fun Operand<*>.asOfAny(): Operand<Any> = this as Operand<Any>
fun Operand<*>.asOfNumber(): Operand<Number> = this as Operand<Number>