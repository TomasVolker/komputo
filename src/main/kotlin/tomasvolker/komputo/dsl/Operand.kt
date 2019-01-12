package tomasvolker.komputo.dsl

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Shape
import tomasvolker.komputo.toIntArray1D
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D

val <T> Operand<T>.name: String get() = asOutput().op().name()
val <T> Operand<T>.localName: String get() = asOutput().op().name().substringAfterLast('/')
val <T> Operand<T>.shape: IntArray1D get() = asOutput().shape().toIntArray1D()
val <T> Operand<T>.dataType: DataType get() = asOutput().dataType()


