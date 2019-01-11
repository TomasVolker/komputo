package tomasvolker.komputo.performance

open class Timer(
    var nanos: Long = System.nanoTime()
) {

    fun reset() { nanos = System.nanoTime() }

    fun tockNanos(): Long = System.nanoTime() - nanos
    fun tickNanos(): Long = tockNanos().also { reset() }

    fun tickSeconds(): Double = tickNanos() / 1e9
    fun tockSeconds(): Double = tockNanos() / 1e9

    companion object: Timer()

}