# website = "https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html"
# class_name = "DeviceMergeSort"

# dcub = dcub_template.template(website)
# dcub.func_sig_template = func_sig_template = '''static cudaError_t $FUNC_NAME$(void* d_temp_storage, size_t& temp_storage_bytes, $FUNC_SIG_ARGS$, BinaryOperator compare_op = BinaryOperator::Max, cudaStream_t stream = nullptr, bool debug_synchronous = false);'''
# dcub.func_impl_template = '''cudaError_t $CLASS_NAME$::$FUNC_NAME$(void* d_temp_storage, size_t& temp_storage_bytes, $FUNC_IMPL_ARGS$, BinaryOperator compare_op, cudaStream_t stream, bool debug_synchronous){
#     return op_mapper(compare_op, [&](auto op) {
#         return ::cub::$CLASS_NAME$::$FUNC_NAME$(d_temp_storage, temp_storage_bytes, $INNER_FUNC_INPUT_ARGS$, op, stream, debug_synchronous);
#     });
# }'''

# DeviceMergeSort = Class(class_name)
# DeviceMergeSort.set_template(dcub)

# SortPairs = Func("SortPairs", Ret(),
#                  [
#                     Arg("$T$*", "d_keys"),
#                     Arg("$I$*", "d_items"),
#                     Arg("int", "num_items"),
#                 ])
# SortPairs.set_template(dcub)
# DeviceMergeSort.add_funcs([ SortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# SortPairsCopy = Func("SortPairsCopy", Ret(),
#                     [
#                         Arg("const $T$*", "d_input_keys"),
#                         Arg("const $I$*", "d_input_items"),
#                         Arg("$T$*", "d_output_keys"),
#                         Arg("$I$*", "d_output_items"),
#                         Arg("int", "num_items"),
#                     ])
# SortPairsCopy.set_template(dcub)
# DeviceMergeSort.add_funcs([ SortPairsCopy.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# SortKeys = Func("SortKeys", Ret(),
#                 [
#                     Arg("$T$*", "d_keys"),
#                     Arg("int", "num_items"),
#                 ])

# SortKeys.set_template(dcub)
# DeviceMergeSort.add_funcs([ SortKeys.instantiate([("$T$", t)]) for t in value_types])

# SortKeysCopy = Func("SortKeysCopy", Ret(),
#                     [
#                         Arg("const $T$*", "d_input_keys"),
#                         Arg("$T$*", "d_output_keys"),
#                         Arg("int", "num_items"),
#                     ])
# SortKeysCopy.set_template(dcub)
# DeviceMergeSort.add_funcs([ SortKeysCopy.instantiate([("$T$", t)]) for t in value_types])


# StableSortPairs = Func("StableSortPairs", Ret(),
#                           [
#                                 Arg("$T$*", "d_keys"),
#                                 Arg("$I$*", "d_items"),
#                                 Arg("int", "num_items"),
#                             ])
# StableSortPairs.set_template(dcub)
# DeviceMergeSort.add_funcs([ StableSortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])
 
# StableSortKeys = Func("StableSortKeys", Ret(),
#                             [
#                                 Arg("$T$*", "d_keys"),
#                                 Arg("int", "num_items"),
#                             ])
# StableSortKeys.set_template(dcub)
# DeviceMergeSort.add_funcs([ StableSortKeys.instantiate([("$T$", t)]) for t in value_types])

# DeviceMergeSort.write(src_ext=".cu", folder="../private/dcub/")

# # LCUB

# lcub = lcub_template.template(website)
# DeviceMergeSort = Class(class_name)
# DeviceMergeSort.set_template(lcub)

# ar = Arg(user_convert="raw")

# SortPairs = Func("SortPairs", Ret(),
#                     [
#                         ar.clone("BufferView<$T$>", "d_keys"),
#                         ar.clone("BufferView<$I$>", "d_items"),
#                         ar.clone("int", "num_items"),
#                         ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                     ])
# SortPairs.set_template(lcub)
# DeviceMergeSort.add_funcs([ SortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# SortPairsCopy = Func("SortPairsCopy", Ret(),
#                     [
#                         ar.clone("BufferView<$T$>", "d_input_keys"),
#                         ar.clone("BufferView<$I$>", "d_input_items"),
#                         ar.clone("BufferView<$T$>", "d_output_keys"),
#                         ar.clone("BufferView<$I$>", "d_output_items"),
#                         ar.clone("int", "num_items"),
#                         ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                     ])
# SortPairsCopy.set_template(lcub)
# DeviceMergeSort.add_funcs([ SortPairsCopy.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# SortKeys = Func("SortKeys", Ret(),
#                 [
#                     ar.clone("BufferView<$T$>", "d_keys"),
#                     ar.clone("int", "num_items"),
#                     ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                 ])
# SortKeys.set_template(lcub)
# DeviceMergeSort.add_funcs([ SortKeys.instantiate([("$T$", t)]) for t in value_types])

# SortKeysCopy = Func("SortKeysCopy", Ret(),
#                     [
#                         ar.clone("BufferView<$T$>", "d_input_keys"),
#                         ar.clone("BufferView<$T$>", "d_output_keys"),
#                         ar.clone("int", "num_items"),
#                         ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                     ])

# SortKeysCopy.set_template(lcub)
# DeviceMergeSort.add_funcs([ SortKeysCopy.instantiate([("$T$", t)]) for t in value_types])

# StableSortPairs = Func("StableSortPairs", Ret(),
#                             [
#                                 ar.clone("BufferView<$T$>", "d_keys"),
#                                 ar.clone("BufferView<$I$>", "d_items"),
#                                 ar.clone("int", "num_items"),
#                                 ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                             ])
# StableSortPairs.set_template(lcub)
# DeviceMergeSort.add_funcs([ StableSortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# StableSortKeys = Func("StableSortKeys", Ret(),
#                             [
#                                 ar.clone("BufferView<$T$>", "d_keys"),
#                                 ar.clone("int", "num_items"),
#                                 ar.clone("dcub::BinaryOperator", "compare_op", default_value="dcub::BinaryOperator::Max"),
#                             ])
# StableSortKeys.set_template(lcub)
# DeviceMergeSort.add_funcs([ StableSortKeys.instantiate([("$T$", t)]) for t in value_types])

# DeviceMergeSort.write(folder="../")

from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_histogram.html"
class_name = "DeviceHistogram"

DeviceHistogram = Class(class_name)
dcub = dcub_template.template(website)
DeviceHistogram.set_template(dcub)

 