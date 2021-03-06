package tomasvolker.komputo.dsl

import org.tensorflow.Tensor
import tomasvolker.numeriko.core.implementations.numeriko.array0d.double.NumerikoDoubleArray0D
import tomasvolker.numeriko.core.implementations.numeriko.array0d.generic.NumerikoArray0D
import tomasvolker.numeriko.core.implementations.numeriko.array1d.double.NumerikoDoubleArray1D
import tomasvolker.numeriko.core.implementations.numeriko.array1d.generic.NumerikoArray1D
import tomasvolker.numeriko.core.implementations.numeriko.array2d.double.NumerikoDoubleArray2D
import tomasvolker.numeriko.core.implementations.numeriko.arraynd.NumerikoArrayND
import tomasvolker.numeriko.core.implementations.numeriko.arraynd.NumerikoDoubleArrayND
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.arraynd.generic.ArrayND
import tomasvolker.numeriko.core.interfaces.factory.*
import java.nio.FloatBuffer
import java.nio.IntBuffer

fun Tensor<*>.toDoubleNDArray(): DoubleArrayND {

    val buffer = FloatArray(numElements())
    writeTo(FloatBuffer.wrap(buffer))

    val data = DoubleArray(buffer.size) { i -> buffer[i].toDouble() }

    val shape = shape().map { it.toInt() }.toIntArray()

    return when(numDimensions()) {
        0 -> doubleArray0D(floatValue().toDouble())
        1 -> doubleArray1D(data)
        2 -> doubleArray2D(shape[0], shape[1], data)
        else -> {
            doubleArrayND(
                shape.toIntArray1D(),
                data
            )
        }
    }
}


fun ArrayND<*>.toTensor(): Tensor<*> = when(this) {
    is DoubleArrayND -> toTensor()
    is NumerikoArray1D<*> -> {

        if (first() is String) {
            this as NumerikoArray1D<String>
            Tensor.create(data.map { it.toByteArray(Charsets.US_ASCII) })
        } else
            TODO("unsupported")
    }
    is NumerikoArray0D<*> -> {
        val value = getValue()
        if (value is String) {
            Tensor.create(value.toByteArray(Charsets.US_ASCII))
        } else
            TODO("unsupported")
    }
    else -> TODO("unsupported")
}


// TODO improve numeriko performance
fun DoubleArrayND.toTensor(): Tensor<*> = when(this) {
    is NumerikoDoubleArray0D -> Tensor.create(
        longArrayOf(),
        FloatBuffer.wrap(floatArrayOf(data.toFloat()))
    )
    is NumerikoDoubleArray1D -> Tensor.create(
        longArrayOf(size.toLong()),
        FloatBuffer.wrap(
            FloatArray(size) { i -> data[i].toFloat() }
        )
    )
    is NumerikoDoubleArray2D -> Tensor.create(
        longArrayOf(shape0.toLong(), shape1.toLong()),
        FloatBuffer.wrap(
            FloatArray(size) { i -> data[i].toFloat() }
        )
    )
    is NumerikoDoubleArrayND -> Tensor.create(
        LongArray(rank) { i -> shape(i).toLong() },
        FloatBuffer.wrap(
            FloatArray(size) { i -> data[i].toFloat() }
        )
    )
    else -> Tensor.create(
        LongArray(rank) { i -> shape(i).toLong() },
        FloatBuffer.wrap(
            linearView().let { FloatArray(it.size) { i -> it[i].toFloat() } }
        )
    )
}


fun DoubleArrayND.toIntTensor(): Tensor<*> =
    Tensor.create(
        LongArray(rank) { i -> shape(i).toLong() },
        IntBuffer.wrap(
            linearView().let { IntArray(it.size) { i -> it[i].toInt() } }
        )
    )