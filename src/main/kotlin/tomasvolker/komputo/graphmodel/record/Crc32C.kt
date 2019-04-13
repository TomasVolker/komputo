package tomasvolker.komputo.graphmodel.record

import java.util.zip.Checksum

private val MASK_DELTA: Int = 0xa282ead8.toInt()

fun maskedCrc32c(data: ByteArray) = maskedCrc32c(data, 0, data.size)

fun maskedCrc32c(data: ByteArray, offset: Int, length: Int): Int {
    val crc32c = Crc32C()
    crc32c.update(data, offset, length)
    return crc32c.getMaskedValue()
}

/**
 * Return a masked representation of crc.
 * <p>
 *  Motivation: it is problematic to compute the CRC of a string that
 *  contains embedded CRCs.  Therefore we recommend that CRCs stored
 *  somewhere (e.g., in files) should be masked before being stored.
 * </p>
 * @param crc CRC
 * @return masked CRC
 */
fun mask(crc: Int) = ((crc ushr 15) or (crc shl 17)) + MASK_DELTA;

/**
 * Return the crc whose masked representation is masked_crc.
 * @param maskedCrc masked CRC
 * @return crc whose masked representation is masked_crc
 */
fun unmask(maskedCrc: Int): Int {
    val rot = maskedCrc - MASK_DELTA
    return ((rot ushr 17) or (rot shl 15))
}

class Crc32C: Checksum {

    private val crc32C = PureJavaCrc32C()

    fun getMaskedValue() = mask(getIntValue());

    fun getIntValue() = value.toInt()

    override fun update(b: Int) {
        crc32C.update(b)
    }

    override fun update(b: ByteArray, off: Int, len: Int) {
        crc32C.update(b, off, len);
    }

    override fun getValue() = crc32C.value;

    override fun reset() = crc32C.reset()

}