package tomasvolker.komputo.graphmodel.utils

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

fun ByteArray.asByteBuffer(order: ByteOrder = ByteOrder.nativeOrder()): ByteBuffer =
        ByteBuffer.wrap(this).order(order)

fun ByteArray.asFloatBuffer(): FloatBuffer = asByteBuffer().asFloatBuffer()

fun File.outputStream(append: Boolean): FileOutputStream = FileOutputStream(this, append)
