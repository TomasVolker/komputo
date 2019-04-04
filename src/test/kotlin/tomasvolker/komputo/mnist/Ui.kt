package tomasvolker.komputo.mnist

import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.ColorBuffer
import org.openrndr.draw.colorBuffer
import org.openrndr.math.Vector2
import org.openrndr.math.transforms.transform
import tomasvolker.komputo.dataset.LabeledData
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.interfaces.array2d.generic.Array2D
import tomasvolker.numeriko.core.interfaces.array2d.generic.forEachIndex
import tomasvolker.numeriko.core.interfaces.array2d.generic.get
import tomasvolker.numeriko.core.interfaces.slicing.get
import tomasvolker.numeriko.core.performance.forEach


fun showMnist(image: DoubleArray2D) {

    application {

        configure {

            width = 400
            height = 400
            windowResizable = true

        }

        program {

            val buffer = colorBuffer(image.shape0, image.shape1)

            buffer.write(image)

            extend {

                drawer.model = transform {
                    scale(width.toDouble() / image.shape0)
                }

                drawer.image(buffer)

            }

        }

    }


}

fun ColorBuffer.write(
    image: DoubleArray2D,
    normalize: Boolean = true
) {

    val min = if (normalize) image.min() ?: 0.0 else 0.0
    val max = if (normalize) image.max() ?: 0.0 else 1.0

    fun rescale(x: Double) = (x - min) / (max - min)

    shadow.buffer.rewind()

    forEach(image.shape0, image.shape1) { x, y ->
        shadow[x, y] = ColorRGBa.WHITE.shade(image[x, y])
    }

    shadow.upload()
}


fun showImageMatrix(matrix: Array2D<DoubleArray2D>) {

    application {

        configure {

            width = 400
            height = 400
            windowResizable = true

        }

        program {

            val buffer = colorBuffer(28, 28)

            extend {

                matrix.forEachIndex { i0, i1 ->

                    buffer.write(matrix[i0, i1])

                    val side = width / 10.0

                    drawer.image(
                        buffer,
                        position = Vector2(i0 * side, i1 * side),
                        width = side,
                        height = side
                    )

                }

            }

        }

    }

}