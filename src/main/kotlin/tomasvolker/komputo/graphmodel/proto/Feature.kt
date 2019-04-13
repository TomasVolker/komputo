package tomasvolker.komputo.graphmodel.proto

import com.google.protobuf.ByteString
import tomasvolker.komputo.graphmodel.utils.toByteString
import org.tensorflow.example.*

fun feature(block: Feature.Builder.()->Unit): Feature =
        Feature.newBuilder().apply(block).build()

fun byteList(block: BytesList.Builder.()->Unit): BytesList =
        BytesList.newBuilder().apply(block).build()

fun floatList(block: FloatList.Builder.()->Unit): FloatList =
        FloatList.newBuilder().apply(block).build()

fun int64List(block: Int64List.Builder.()->Unit): Int64List =
        Int64List.newBuilder().apply(block).build()

fun exampleProto(block: Example.Builder.()->Unit): Example =
        Example.newBuilder().apply(block).build()

fun Example.Builder.features(block: Features.Builder.()->Unit): Features =
        featuresProto(block).also { features = it }

fun featuresProto(block: Features.Builder.()->Unit): Features =
        Features.newBuilder().apply(block).build()

fun Features.Builder.feature(name: String, block: Feature.Builder.()->Unit) {
    putFeature(name, feature(block))
}

fun Feature.Builder.byteList(vararg values: ByteString) {
        bytesList = byteList {
            values.forEach { addValue(it) }
        }
    }

fun Feature.Builder.int64List(vararg values: Int) {
    int64List = int64List {
        values.forEach { addValue(it.toLong()) }
    }
}

fun Feature.Builder.int64List(vararg values: Long) {
    int64List = int64List {
        values.forEach { addValue(it) }
    }
}

fun Feature.Builder.floatList(vararg values: Float) {
    floatList = floatList {
        values.forEach { addValue(it) }
    }
}

fun Feature.Builder.floatList(vararg values: Double) {
    floatList = floatList {
        values.forEach { addValue(it.toFloat()) }
    }
}

fun Features.Builder.feature(name: String, value: String) =
        feature(name) { byteList(value.toByteString()) }

fun Features.Builder.feature(name: String, value: ByteArray) =
        feature(name) { byteList(value.toByteString()) }

fun Features.Builder.feature(name: String, value: ByteString) =
        feature(name) { byteList(value) }

fun Features.Builder.feature(name: String, value: Int) =
        feature(name) { int64List(value) }

fun Features.Builder.feature(name: String, value: Long) =
        feature(name) { int64List(value) }

fun Features.Builder.feature(name: String, value: Float) =
        feature(name) { floatList(value) }

fun Features.Builder.feature(name: String, value: Double) =
        feature(name) { floatList(value) }

fun main() {

    exampleProto {

        features {
            feature("planesweep", "hola")
        }

    }.also { println(it) }

}