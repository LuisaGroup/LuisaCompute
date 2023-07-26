from wrapper import *
from dcub_template import value_types, id_type, integer_value_types
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html"
class_name = "DeviceRunLengthEncode"

dcub = dcub_template.template(website)
DeviceRunLengthEncode = Class(class_name)
DeviceRunLengthEncode.set_template(dcub)


# InputIteratorT 	d_in,
# UniqueOutputIteratorT 	d_unique_out,
# LengthsOutputIteratorT 	d_counts_out,
# NumRunsOutputIteratorT 	d_num_runs_out,
# int 	num_items,
Encode = Func("Encode", Ret(), 
              [Arg("const $T$*", "d_in"), 
               Arg("$T$*", "d_unique_out"), 
               Arg("$I$*", "d_counts_out"), 
               Arg("$I$*", "d_num_runs_out"), 
               Arg("int", "num_items")])
Encode.set_template(dcub)

# InputIteratorT 	d_in,
# OffsetsOutputIteratorT 	d_offsets_out,
# LengthsOutputIteratorT 	d_lengths_out,
# NumRunsOutputIteratorT 	d_num_runs_out,
# int 	num_items,
NonTrivialRuns = Func("NonTrivialRuns", Ret(), 
                      [Arg("const $T$*", "d_in"), 
                       Arg("$I$*", "d_offsets_out"), 
                       Arg("$I$*", "d_lengths_out"), 
                       Arg("$I$*", "d_num_runs_out"), 
                       Arg("int", "num_items")])
NonTrivialRuns.set_template(dcub)

DeviceRunLengthEncode.add_funcs([ Encode.instantiate([("$T$", t),("$I$", "int32_t")]) for t in integer_value_types])
DeviceRunLengthEncode.add_funcs([ NonTrivialRuns.instantiate([("$T$", t),("$I$", "int32_t")]) for t in integer_value_types])

DeviceRunLengthEncode.write(src_ext=".cu", folder="../private/dcub/")


# LCUB
lcub = lcub_template.template(website)
DeviceRunLengthEncode = Class(class_name)
DeviceRunLengthEncode.set_template(lcub)

ar = Arg(user_convert="raw")
Encode = Func("Encode", Ret(), 
              [ar.clone("BufferView<$T$>", "d_in"), 
               ar.clone("BufferView<$T$>", "d_unique_out"), 
               ar.clone("BufferView<$I$>", "d_counts_out"), 
               ar.clone("BufferView<$I$>", "d_num_runs_out"), 
               ar.clone("int", "num_items")])
Encode.set_template(lcub)

NonTrivialRuns = Func("NonTrivialRuns", Ret(),
                        [ar.clone("BufferView<$T$>", "d_in"),
                         ar.clone("BufferView<$I$>", "d_offsets_out"),
                         ar.clone("BufferView<$I$>", "d_lengths_out"),
                         ar.clone("BufferView<$I$>", "d_num_runs_out"),
                         ar.clone("int", "num_items")])
NonTrivialRuns.set_template(lcub)

DeviceRunLengthEncode.add_funcs([ Encode.instantiate([("$T$", t),("$I$", "int32_t")]) for t in integer_value_types])
DeviceRunLengthEncode.add_funcs([ NonTrivialRuns.instantiate([("$T$", t),("$I$", "int32_t")]) for t in integer_value_types])

DeviceRunLengthEncode.write(folder="../")
