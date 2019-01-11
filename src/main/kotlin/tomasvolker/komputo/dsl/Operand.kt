package tomasvolker.tensorflow.dsl

import org.tensorflow.DataType
import org.tensorflow.Operand
import org.tensorflow.Shape

val <T> Operand<T>.name: String get() = asOutput().op().name()
val <T> Operand<T>.shape: Shape get() = asOutput().shape()
val <T> Operand<T>.dataType: DataType get() = asOutput().dataType()


