package tomasvolker.komputo.graphmodel.record


import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun InputStream.tfRecordReader(crcCheck: Boolean = true) = TFRecordReader(this, crcCheck)
fun File.tfRecordReader(bufferSize: Int = DEFAULT_BUFFER_SIZE) = inputStream().buffered(bufferSize).tfRecordReader()

class TFRecordReader(
        val input: InputStream,
        val crcCheck: Boolean
): Closeable by input {

    fun read(): ByteArray? {
        /**
         * TFRecord format:
         * uint64 length
         * uint32 masked_crc32_of_length
         * byte   data[length]
         * uint32 masked_crc32_of_data
         */
        val lenBytes = ByteArray(8)
        try {
            // Only catch EOF here, other case means corrupted file
            readFully(input, lenBytes)
        } catch (eof: EOFException) {
            return null // return null means EOF
        }

        val len = fromInt64LE(lenBytes)

        // Verify length crc32
        if (!crcCheck) {
            input.skip(4)
        } else {
            val lenCrc32Bytes = ByteArray(4)
            readFully(input, lenCrc32Bytes)
            val lenCrc32 = fromInt32LE(lenCrc32Bytes)
            if (lenCrc32 != maskedCrc32c(lenBytes)) {
                throw IOException("Length header crc32 checking failed: " + lenCrc32 + " != " + maskedCrc32c(lenBytes) + ", length = " + len)
            }
        }

        if (len > Integer.MAX_VALUE) {
            throw IOException("Record size exceeds max value of int32: $len")
        }
        val data = ByteArray(len.toInt())
        readFully(input, data)

        // Verify data crc32
        if (!crcCheck) {
            input.skip(4)
        } else {
            val dataCrc32Bytes = ByteArray(4)
            readFully(input, dataCrc32Bytes)
            val dataCrc32 = fromInt32LE(dataCrc32Bytes)
            if (dataCrc32 != maskedCrc32c(data)) {
                throw IOException("Data crc32 checking failed: " + dataCrc32 + " != " + maskedCrc32c(data))
            }
        }
        return data
    }

    private fun fromInt64LE(data: ByteArray): Long {
        assert(data.size == 8)
        val bb = ByteBuffer.wrap(data)
        bb.order(ByteOrder.LITTLE_ENDIAN)
        return bb.long
    }

    private fun fromInt32LE(data: ByteArray): Int {
        assert(data.size == 4)
        val bb = ByteBuffer.wrap(data)
        bb.order(ByteOrder.LITTLE_ENDIAN)
        return bb.int
    }

    @Throws(IOException::class)
    private fun readFully(`in`: InputStream, buffer: ByteArray) {
        var nbytes: Int
        var nread = 0
        while (nread < buffer.size) {
            nbytes = `in`.read(buffer, nread, buffer.size - nread)
            if (nbytes < 0) {
                throw EOFException("End of file reached before reading fully.")
            }
            nread += nbytes
        }
    }
}