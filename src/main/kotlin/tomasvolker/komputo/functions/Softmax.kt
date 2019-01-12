package tomasvolker.komputo.functions

import tomasvolker.numeriko.core.interfaces.array1d.double.DoubleArray1D
import tomasvolker.numeriko.core.interfaces.array1d.double.applyElementWise
import tomasvolker.numeriko.core.interfaces.array1d.double.elementWise
import kotlin.math.exp

fun softmax(array: DoubleArray1D): DoubleArray1D {
    val result = array.elementWise { exp(it) }.asMutable()
    val sum = result.sum()
    result.applyElementWise { it / sum }
    return result
}
