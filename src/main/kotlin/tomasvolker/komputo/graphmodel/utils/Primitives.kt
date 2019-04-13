package tomasvolker.komputo.graphmodel.utils

import com.google.protobuf.ByteString
import java.nio.charset.Charset

fun String.toByteString(charset: Charset = Charsets.UTF_8): ByteString {
    val output = ByteString.newOutput()
    output.writer(charset).use {
        it.write(this)
    }
    return output.toByteString()
}
