package tomasvolker.komputo.graphmodel

sealed class Scope {

    abstract val fullName: String

    abstract val parentList: List<Scope>

}

object RootScope: Scope() {

    override val fullName: String
        get() = ""

    override val parentList: List<Scope>
        get() = emptyList()

}

class SubScope(
    val name: String,
    val parent: Scope
): Scope() {

    override val fullName: String
        get() = when(parent) {
            is RootScope -> name
            is SubScope -> "${parent.fullName}/$name"
        }

    override val parentList: List<Scope>
        get() = parent.parentList + parent

}
