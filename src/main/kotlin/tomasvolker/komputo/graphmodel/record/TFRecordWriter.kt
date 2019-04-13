package tomasvolker.komputo.graphmodel.record

import com.google.protobuf.MessageLite
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun OutputStream.tfRecordWriter() = TFRecordWriter(this)

fun File.tfRecordWriter(bufferSize: Int = DEFAULT_BUFFER_SIZE) =
        outputStream().buffered(bufferSize).tfRecordWriter()

fun TFRecordWriter.write(message: MessageLite) {
    write(message.toByteArray())
}

class TFRecordWriter(
        val output: OutputStream
): Closeable by output {

    fun write(record: ByteArray, offset: Int = 0, length: Int = record.size) {
        /*
         * TFRecord format:
         * uint64 length
         * uint32 masked_crc32_of_length
         * byte   data[length]
         * uint32 masked_crc32_of_data
         */
        val len = length.toLong().toInt64LE()

        output.run {
            write(len)
            write(maskedCrc32c(len).toInt32LE())
            write(record, offset, length)
            write(maskedCrc32c(record, offset, length).toInt32LE())
        }

    }

    private fun Long.toInt64LE(): ByteArray {
        val buff = ByteArray(8)
        val bb = ByteBuffer.wrap(buff)
        bb.order(ByteOrder.LITTLE_ENDIAN)
        bb.putLong(this)
        return buff
    }

    private fun Int.toInt32LE(): ByteArray {
        val buff = ByteArray(4)
        val bb = ByteBuffer.wrap(buff)
        bb.order(ByteOrder.LITTLE_ENDIAN)
        bb.putInt(this)
        return buff
    }
}