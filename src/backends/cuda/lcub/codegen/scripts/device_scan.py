from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_scan.html"
class_name = "DeviceScan"


DeviceScan = Class(class_name)
dcub = dcub_template.template(website)
DeviceScan.set_template(dcub)

TEMP = Func("", Ret(), [Arg("const $T$*", "d_in"), Arg("$T$*", "d_out"), Arg("int", "num_items")])
TEMP.set_template(dcub)

ExclusiveSum = TEMP.rename("ExclusiveSum")
InclusiveSum = TEMP.rename("InclusiveSum")

TEMP = Func("", Ret(), 
            [
                Arg("const $I$*", "d_keys_in"), 
                Arg("const $T$*", "d_values_in"), 
                Arg("$T$*","d_values_out"), 
                Arg("int", "num_items"), 
                Arg(auto_fill="::cub::Equality{}")
            ])
TEMP.set_template(dcub)
ExclusiveSumByKey = TEMP.rename("ExclusiveSumByKey")
InclusiveSumByKey = TEMP.rename("InclusiveSumByKey")

DeviceScan.add_funcs([ ExclusiveSum.instantiate([("$T$", t)]) for t in value_types])
DeviceScan.add_funcs([ InclusiveSum.instantiate([("$T$", t)]) for t in value_types])
DeviceScan.add_funcs([ ExclusiveSumByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])
DeviceScan.add_funcs([ InclusiveSumByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

DeviceScan.write(src_ext=".cu", folder="../private/dcub/")


# LCUB
DeviceScan = Class(class_name)
lcub = lcub_template.template(website)
DeviceScan.set_template(lcub)

ar = Arg(user_convert="raw")

TEMP = Func("", Ret(), [ar.clone("BufferView<$T$>", "d_in"), ar.clone("BufferView<$T$>", "d_out"), ar.clone("int", "num_items")])
TEMP.set_template(lcub)
# print(TEMP.func_sig())

ExclusiveSum = TEMP.rename("ExclusiveSum")
InclusiveSum = TEMP.rename("InclusiveSum")

TEMP = Func("", Ret(),  
            [ 
                ar.clone("BufferView<$I$>", "d_keys_in"),
                ar.clone("BufferView<$T$>", "d_values_in"),
                ar.clone("BufferView<$T$>", "d_values_out"),
                ar.clone("int", "num_items")
            ])
TEMP.set_template(lcub)

ExclusiveSumByKey = TEMP.rename("ExclusiveSumByKey")
InclusiveSumByKey = TEMP.rename("InclusiveSumByKey")

# print(ExclusiveSumByKey.func_sig())

DeviceScan.add_funcs([ ExclusiveSum.instantiate([("$T$", t)]) for t in value_types])
DeviceScan.add_funcs([ InclusiveSum.instantiate([("$T$", t)]) for t in value_types])
DeviceScan.add_funcs([ ExclusiveSumByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])
DeviceScan.add_funcs([ InclusiveSumByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

DeviceScan.write(folder="../")