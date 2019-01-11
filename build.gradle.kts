
plugins {
    kotlin("jvm") version "1.3.11"
}

group = "tomasvolker"
version = "0.0.1"

repositories {
    mavenCentral()
    maven { url = uri("http://dl.bintray.com/tomasvolker/maven") }
}

dependencies {
    compile(kotlin("stdlib-jdk8"))
    testCompile("junit", "junit", "4.12")

    api(group = "org.tensorflow", name = "tensorflow", version = "1.11.0")
    //api(group = "org.tensorflow", name = "libtensorflow_jni_gpu", version = "1.11.0")

    api(group = "tomasvolker", name = "numeriko-core", version = "0.0.2")
    testImplementation(group = "tomasvolker", name = "kyplot", version = "0.0.1")

}
