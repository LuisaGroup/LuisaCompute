from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html"
class_name = "DeviceRadixSort"

dcub = dcub_template.template(website)
DeviceRadixSort = Class(class_name)
DeviceRadixSort.set_template(dcub)

# template<typename KeyT , typename ValueT , typename NumItemsT >
# static CUB_RUNTIME_FUNCTION
# cudaError_t 	SortPairs (void *d_temp_storage, size_t &temp_storage_bytes, 
# const KeyT *d_keys_in, KeyT *d_keys_out, 
# const ValueT *d_values_in, ValueT *d_values_out, 
# NumItemsT num_items, 
# int begin_bit=0, 
# int end_bit=sizeof(KeyT)*8, cudaStream_t stream=0, bool debug_synchronous=false)

TEMP = Func("", Ret(),
            [
                Arg("const $I$*", "d_keys_in"), 
                Arg("$I$*", "d_keys_out"), 
                Arg("const $T$*", "d_values_in"), 
                Arg("$T$*", "d_values_out"), 
                Arg("int", "num_items"),
                Arg("int", "begin_bit","0"),
                Arg("int", "end_bit", "sizeof($I$)*8"),
            ])

TEMP.set_template(dcub)
SortPairs = TEMP.rename("SortPairs")
SortPairsDescending = TEMP.rename("SortPairsDescending")

DeviceRadixSort.add_funcs([ SortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])
DeviceRadixSort.add_funcs([ SortPairsDescending.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# template<typename KeyT , typename NumItemsT >
# static CUB_RUNTIME_FUNCTION
# cudaError_t 	SortKeys (void *d_temp_storage, size_t &temp_storage_bytes, 
# const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items, int begin_bit=0, 
# int end_bit=sizeof(KeyT)*8, cudaStream_t stream=0, bool debug_synchronous=false)

TEMP = Func("", Ret(), 
            [
                Arg("const $I$*", "d_keys_in"),
                Arg("$I$*", "d_keys_out"),
                Arg("int", "num_items"),
                Arg("int", "begin_bit","0"),
                Arg("int", "end_bit","sizeof($I$)*8"),
            ])
TEMP.set_template(dcub)

SortKeys = TEMP.rename("SortKeys")
SortKeysDescending = TEMP.rename("SortKeysDescending")
DeviceRadixSort.add_funcs([ SortKeys.instantiate([("$I$", id_type)])])
DeviceRadixSort.add_funcs([ SortKeysDescending.instantiate([("$I$", id_type)])])

DeviceRadixSort.write(src_ext=".cu", folder="../private/dcub/")

lcub = lcub_template.template(website)
DeviceRadixSort = Class(class_name)
DeviceRadixSort.set_template(lcub)

ar = Arg(user_convert="raw")

TEMP = Func("", Ret(),
            [
                ar.clone("BufferView<$I$>", "d_keys_in"),
                ar.clone("BufferView<$I$>", "d_keys_out"),
                ar.clone("BufferView<$T$>", "d_values_in"),
                ar.clone("BufferView<$T$>", "d_values_out"),
                ar.clone("int", "num_items"),
                ar.clone("int", "begin_bit","0"),
                ar.clone("int", "end_bit","sizeof($I$)*8"),
            ])
TEMP.set_template(lcub)

SortPairs = TEMP.rename("SortPairs")
SortPairsDescending = TEMP.rename("SortPairsDescending")

DeviceRadixSort.add_funcs([ SortPairs.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])
DeviceRadixSort.add_funcs([ SortPairsDescending.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])


TEMP = Func("", Ret(), 
            [
                ar.clone("const $I$*", "d_keys_in"),
                ar.clone("$I$*", "d_keys_out"),
                ar.clone("int", "num_items"),
                ar.clone("int", "begin_bit","0"),
                ar.clone("int", "end_bit","sizeof($I$)*8"),
            ])
TEMP.set_template(lcub)

SortKeys = TEMP.rename("SortKeys")
SortKeysDescending = TEMP.rename("SortKeysDescending")

DeviceRadixSort.add_funcs([ SortKeys.instantiate([("$I$", id_type)])])
DeviceRadixSort.add_funcs([ SortKeysDescending.instantiate([("$I$", id_type)])])

DeviceRadixSort.write(folder="../")
