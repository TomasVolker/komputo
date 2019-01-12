package tomasvolker.komputo.builder

import tomasvolker.komputo.TFOperand
import tomasvolker.komputo.dsl.group

fun ModelBuilder.group(operations: Iterable<TFOperand>): TFOperand = ops.group(operations)
fun ModelBuilder.group(vararg operations: TFOperand): TFOperand = ops.group(*operations)

fun ModelBuilder.group(name: String, vararg operations: TFOperand): TFOperand = ops.group(name, *operations)
fun ModelBuilder.group(name: String, operations: Iterable<TFOperand>): TFOperand = ops.group(name, operations)