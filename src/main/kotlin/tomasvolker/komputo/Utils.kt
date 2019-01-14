package tomasvolker.komputo

import org.tensorflow.*
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import org.tensorflow.types.UInt8
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.intArray1D
import tomasvolker.numeriko.core.interfaces.factory.intArray1DOf
import java.io.File
import java.lang.IllegalArgumentException

typealias TFOperand = Operand<*>
typealias TFVariable = Variable<*>
typealias TFPlaceholder = Placeholder<*>
typealias TFConstant = Constant<*>

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

val scalar: IntArray1D = intArray1DOf()

fun Shape.toIntArray1D() = intArray1D(numDimensions()) { i -> size(i).toInt() }

fun IntArray1D?.toShape(): Shape =
    when {
        this == null -> Shape.unknown()
        size == 0 -> Shape.scalar()
        else -> Shape.make(
            this[0].toLong(),
            *this.drop(1).map { it.toLong() }.toLongArray()
        )
    }


fun TFOperand.asOfAny(): Operand<Any> = this as Operand<Any>
fun TFOperand.asOfNumber(): Operand<Number> = this as Operand<Number>
fun TFOperand.asOfString(): Operand<String> = this as Operand<String>


fun loadGraphDef(path: String): Graph = loadGraphDef(File(path))
fun loadGraphDef(file: File): Graph = Graph().apply { importGraphDef(file.readBytes()) }

val Operation.outputs: List<Output<*>> get() =
    List(numOutputs()) { i -> output<Any>(i) }

fun Output<*>.safeToString(): String {
    val datatype = try {
        dataType()
    } catch (e: IllegalArgumentException) {
        null
    }
    return "<index = ${this.index()}, shape = ${shape()}, datatype = $datatype>"
}