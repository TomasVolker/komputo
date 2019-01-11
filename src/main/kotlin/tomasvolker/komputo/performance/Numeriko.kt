package tomasvolker.performance

import tomasvolker.numeriko.core.dsl.I
import tomasvolker.numeriko.core.index.All
import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array1d.generic.forEachIndex
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.interfaces.arraynd.double.DoubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.doubleArrayND
import tomasvolker.numeriko.core.interfaces.factory.doubleZeros
import tomasvolker.numeriko.core.interfaces.factory.intArray1D
import java.lang.IllegalArgumentException

fun stack(arrays: List<DoubleArray2D>, axis: Int = 0): DoubleArrayND {

    require(axis in 0..2) { "Stacking axis must be 0 or 1" }

    if (arrays.isEmpty()) return doubleArrayND(I[0, 0, 0]) { 0.0 }

    val resultShape = arrays.first().shape
    require(arrays.all { it.shape == resultShape }) { "All shapes must be the same" }

    return when(axis) {
        0 -> doubleZeros(I[arrays.size, resultShape[0], resultShape[1]]).asMutable().apply {
            forEach(arrays.size, resultShape[0], resultShape[1]) { i0, i1, i2 ->
                setDouble(arrays[i0][i1, i2], i0, i1, i2)
            }
        }
        1 -> doubleZeros(I[resultShape[0], arrays.size, resultShape[1]]).asMutable().apply {
            forEach(resultShape[0], arrays.size, resultShape[1]) { i0, i1, i2 ->
                setDouble(arrays[i0][i1, i2], i0, i1, i2)
            }
        }
        2 -> doubleZeros(I[resultShape[0], resultShape[1], arrays.size]).asMutable().apply {
            forEach(resultShape[0], resultShape[1], arrays.size) { i0, i1, i2 ->
                setDouble(arrays[i0][i1, i2], i0, i1, i2)
            }
        }
        else -> throw IllegalStateException()
    }

}

fun DoubleArray1D.argmax(): Int {
    var resultIndex = 0
    var resultValue = Double.NEGATIVE_INFINITY

    forEachIndex { i ->
        val value = this[i]
        if (value > resultValue) {
            resultIndex = i
            resultValue = value
        }
    }
    return resultIndex
}

fun DoubleArray2D.reduceArgmax(axis: Int = 0): IntArray1D = when(axis) {
    0 -> intArray1D(shape1) { i -> this[All, i].argmax() }
    1 -> intArray1D(shape0) { i -> this[i, All].argmax() }
    else -> throw IllegalArgumentException("axis out of bounds")
}
