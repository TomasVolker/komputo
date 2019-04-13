package tomasvolker.komputo.graphmodel.proto

import com.google.protobuf.ByteString
import java.nio.ByteBuffer

fun ByteArray.toByteString(): ByteString = ByteString.copyFrom(this)
fun ByteBuffer.toByteString(): ByteString = ByteString.copyFrom(this)

fun buildByteString(initialCapacity: Int = 128, block: ByteString.Output.()->Unit): ByteString =
        ByteString.newOutput(initialCapacity).apply(block).toByteString()
