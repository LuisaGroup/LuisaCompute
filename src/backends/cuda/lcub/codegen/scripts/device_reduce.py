from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html"
class_name = "DeviceReduce"

# DeviceReduce
DeviceReduce = Class(class_name)
dcub = dcub_template.template(website)
DeviceReduce.set_template(dcub)

ar = Arg(user_convert="raw")

TEMP = Func("", Ret(), [Arg("const $T$*", "d_in"), Arg("$T$*", "d_out"), Arg("int", "num_items")])
TEMP.set_template(dcub)

Sum = TEMP.rename("Sum")
Max = TEMP.rename("Max")
Min = TEMP.rename("Min")

TEMP = Func("", Ret(), [Arg("const $T$*", "d_in"), 
                 Arg("KeyValuePair<$I$,$T$>*", "d_out","", "::cub::KeyValuePair<$I$,$T$>*"), 
                 Arg("int", "num_items")])
TEMP.set_template(dcub)

ArgMax = TEMP.rename("ArgMax")
ArgMin = TEMP.rename("ArgMin")

DeviceReduce.add_funcs([ Sum.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ Max.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ Min.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ArgMin.instantiate([("$T$", b), ("$I$", id_type)]) for b in value_types])
DeviceReduce.add_funcs([ArgMax.instantiate([("$T$", b), ("$I$", id_type)]) for b in value_types])

# write to files
DeviceReduce.write(src_ext=".cu", folder="../private/dcub/")

# DeviceReduce
DeviceReduce = Class(class_name)
lcub = lcub_template.template(website)
DeviceReduce.set_template(lcub)

ar = Arg(user_convert="raw")

TEMP = Func("", Ret(), [ar.clone("BufferView<$T$>", "d_in"), ar.clone("BufferView<$T$>", "d_out"), ar.clone("int", "num_items")])
TEMP.set_template(lcub)

Sum = TEMP.rename("Sum")
Max = TEMP.rename("Max")
Min = TEMP.rename("Min")

TEMP = Func("", Ret(), [ar.clone("BufferView<$T$>", "d_in"), 
                 ar.clone("BufferView<dcub::KeyValuePair<$I$,$T$>>", "d_out"), 
                 ar.clone("int", "num_items")])
TEMP.set_template(lcub)

ArgMax = TEMP.rename("ArgMax")
ArgMin = TEMP.rename("ArgMin")

DeviceReduce.add_funcs([ Sum.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ Max.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ Min.instantiate([("$T$", t)]) for t in value_types])
DeviceReduce.add_funcs([ArgMin.instantiate([("$T$", b), ("$I$", id_type)]) for b in value_types])
DeviceReduce.add_funcs([ArgMax.instantiate([("$T$", b), ("$I$", id_type)]) for b in value_types])

# write to files
DeviceReduce.write(folder="../")



