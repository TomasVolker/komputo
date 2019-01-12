package tomasvolker.komputo.mnist

import tomasvolker.komputo.dataset.LabeledData
import tomasvolker.komputo.dataset.labelTo
import tomasvolker.numeriko.core.interfaces.array1d.integer.IntArray1D
import tomasvolker.numeriko.core.interfaces.array2d.double.DoubleArray2D
import tomasvolker.numeriko.core.interfaces.array2d.generic.indices0
import tomasvolker.numeriko.core.interfaces.array2d.generic.indices1
import tomasvolker.numeriko.core.interfaces.factory.doubleZeros
import tomasvolker.numeriko.core.interfaces.factory.intArray1D
import java.io.*


fun InputStream.data() = DataInputStream(this)
fun OutputStream.data() = DataOutputStream(this)

inline fun <T :AutoCloseable, R> T.runUsing(block: T.()->R): R = use(block)

object Mnist {

    val LABEL_FILE_MAGIC_NUMBER = 2049
    val IMAGE_FILE_MAGIC_NUMBER = 2051

    fun loadLabels(path: String): IntArray1D = loadLabels(File(path))

    fun loadImages(path: String): List<DoubleArray2D> = loadImages(File(path))

    fun loadDataset(
        imagesPath: String,
        labelsPath: String
    ): List<LabeledData<DoubleArray2D, Int>> = loadDataset(
        File(imagesPath),
        File(labelsPath)
    )

    fun loadDataset(
        imagesFile: File,
        labelsFile: File
    ): List<LabeledData<DoubleArray2D, Int>> =
            loadImages(imagesFile).zip(
                loadLabels(
                    labelsFile
                )
            ) { image, label ->
                image labelTo label
            }

    fun loadLabels(file: File): IntArray1D =
        file.inputStream().buffered().data().runUsing {

            val magicNumber = readInt()

            require(magicNumber == LABEL_FILE_MAGIC_NUMBER) {
                "The file is not a MNIST label file"
            }

            val numLabels = readInt()

            intArray1D(numLabels) { readUnsignedByte() }
        }


    fun loadImages(file: File): List<DoubleArray2D> =
        file.inputStream().buffered().data().runUsing {

            val magicNumber = readInt()

            require(magicNumber == IMAGE_FILE_MAGIC_NUMBER) {
                "The file is not a MNIST data file"
            }

            val imageCount = readInt()
            val height = readInt()
            val width = readInt()

            List(imageCount) {

                doubleZeros(width, height).asMutable().also {

                    for (y in 0 until height) {
                        for (x in 0 until width) {
                            it[x, y] = readUnsignedByte() / 255.0
                        }
                    }

                }
            }
        }


    fun renderToString(image: DoubleArray2D) = buildString {
        for (y in image.indices1) {
            append("|")
            for (x in image.indices0) {

                val pixel = image[x, y]

                val char = when {
                    pixel < 1.0 / 4.0 -> " "
                    pixel < 2.0 / 4.0 -> "."
                    pixel < 3.0 / 4.0 -> "x"
                    else -> "X"
                }

                append(char)
            }
            append("|\n")
        }
    }


}