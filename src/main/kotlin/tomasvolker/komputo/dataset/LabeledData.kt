package tomasvolker.komputo.dataset

data class LabeledData<out D, out L>(
    val data: D,
    val label: L
)

typealias LabeledDataset<D, L> = List<LabeledData<D, L>>

inline fun <D, D2, L> LabeledData<D, L>.mapData(transform: (D)->D2) =
    LabeledData(transform(data), label)

inline fun <D, L, L2> LabeledData<D, L>.mapLabel(transform: (L)->L2) =
    LabeledData(data, transform(label))

infix fun <D, L> D.labelTo(label: L) = LabeledData(this, label)

inline fun <D, D2, L> LabeledDataset<D, L>.mapData(transform: (D)->D2) =
    map { it.mapData(transform) }

inline fun <D, L, L2> LabeledDataset<D, L>.mapLabels(transform: (L)->L2) =
    map { it.mapLabel(transform) }

inline fun <D, L> List<D>.labelEach(labeler: (D)->L): LabeledDataset<D, L> =
    map { it labelTo labeler(it) }
