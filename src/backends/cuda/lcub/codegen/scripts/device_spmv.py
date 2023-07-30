from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_spmv.html"
class_name = "DeviceSpmv"
 
DeviceSpMV = Class(class_name)
dcub = dcub_template.template(website)
DeviceSpMV.set_template(dcub)
 
# template<typename ValueT >
# static CUB_RUNTIME_FUNCTION
# cudaError_t 	CsrMV (void *d_temp_storage, 
# size_t &temp_storage_bytes, 
# 
# const ValueT *d_values, 
# const int *d_row_offsets, 
# const int *d_column_indices, 
# const ValueT *d_vector_x, 
# ValueT *d_vector_y, 
# int num_rows, 
# int num_cols, 
# int num_nonzeros, 
# 
# cudaStream_t stream=0, bool debug_synchronous=false)

CsrMV = Func("CsrMV", Ret(), 
            [
                Arg("const $T$*", "d_values"),
                Arg("const int*", "d_row_offsets"),
                Arg("const int*", "d_column_indices"),
                Arg("const $T$*", "d_vector_x"),
                Arg("$T$*", "d_vector_y"),
                Arg("int", "num_rows"),
                Arg("int", "num_cols"),
                Arg("int", "num_nonzeros"),
            ])
CsrMV.set_template(dcub)

DeviceSpMV.add_funcs([ CsrMV.instantiate([("$T$", t)]) for t in value_types])

DeviceSpMV.write(src_ext=".cu", folder="../private/dcub/")


# LCUB
DeviceSpMV = Class(class_name)
lcub = lcub_template.template(website)
DeviceSpMV.set_template(lcub)
 
ar = Arg(user_convert="raw")

CsrMV = Func("CsrMV", Ret(),
            [
                ar.clone("BufferView<$T$>", "d_values"),
                ar.clone("BufferView<int>", "d_row_offsets"),
                ar.clone("BufferView<int>", "d_column_indices"),
                ar.clone("BufferView<$T$>", "d_vector_x"),
                ar.clone("BufferView<$T$>", "d_vector_y"),
                ar.clone("int", "num_rows"),
                ar.clone("int", "num_cols"),
                ar.clone("int", "num_nonzeros"),
            ])
CsrMV.set_template(lcub)
 
DeviceSpMV.add_funcs([ CsrMV.instantiate([("$T$", t)]) for t in value_types])

DeviceSpMV.write(folder="../")

                