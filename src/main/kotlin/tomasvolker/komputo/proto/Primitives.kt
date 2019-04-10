package tomasvolker.komputo.proto

import com.google.protobuf.ByteString
import java.nio.charset.Charset

fun String.toByteString(charset: Charset = Charsets.UTF_8): ByteString {
    val result = ByteString.newOutput()
    result.writer(charset).write(this)
    return result.toByteString()
}

fun ByteArray.toByteString(): ByteString = ByteString.copyFrom(this)
