package tomasvolker.komputo.performance

inline fun forEach(
    times0: Int,
    block: (Int)->Unit
) {
    for (i0 in 0 until times0) {
        block(i0)
    }
}

inline fun forEach(
    times0: Int,
    times1: Int,
    block: (Int, Int)->Unit
) {
    forEach(times0) { i0 ->
        forEach(times1) { i1 ->
            block(i0, i1)
        }
    }
}

inline fun forEach(
    times0: Int,
    times1: Int,
    times2: Int,
    block: (Int, Int, Int)->Unit
) {
    forEach(times0, times1) { i0, i1 ->
        forEach(times2) { i2 ->
            block(i0, i1, i2)
        }
    }
}

inline fun forEach(
    times0: Int,
    times1: Int,
    times2: Int,
    times3: Int,
    block: (Int, Int, Int, Int)->Unit
) {
    forEach(times0, times1, times2) { i0, i1, i2 ->
        forEach(times3) { i3 ->
            block(i0, i1, i2, i3)
        }
    }
}