from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_adjacent_difference.html"
class_name = "DeviceAdjacentDifference"

# DCUB

dcub = dcub_template.template(website)
DeviceAdjacentDifference = Class(class_name)
DeviceAdjacentDifference.set_template(dcub)

TEMP = Func("", Ret(),
            [
                Arg("const $T$*", "d_input"),
                Arg("$T$*", "d_output"),
                Arg("int", "num_items"),
                Arg(auto_fill = "Difference{}")
            ])
TEMP.set_template(dcub)

SubtractLeftCopy = TEMP.rename("SubtractLeftCopy")
SubtractRightCopy = TEMP.rename("SubtractRightCopy")

DeviceAdjacentDifference.add_funcs([SubtractLeftCopy.instantiate([("$T$", t)]) for t in value_types])
DeviceAdjacentDifference.add_funcs([SubtractRightCopy.instantiate([("$T$", t)]) for t in value_types])

TEMP = Func("", Ret(),
            [
                Arg("$T$*", "d_input"),
                Arg("int", "num_items"),
                Arg(auto_fill = "Difference{}")
            ])
TEMP.set_template(dcub)

SubtractLeft = TEMP.rename("SubtractLeft")
SubtractRight = TEMP.rename("SubtractRight")

DeviceAdjacentDifference.add_funcs([SubtractLeft.instantiate([("$T$", t)]) for t in value_types])
DeviceAdjacentDifference.add_funcs([SubtractRight.instantiate([("$T$", t)]) for t in value_types])

DeviceAdjacentDifference.write(src_ext=".cu", folder="../private/dcub/")

# LCUB

lcub = lcub_template.template(website)
DeviceAdjacentDifference = Class(class_name) 
DeviceAdjacentDifference.set_template(lcub)

ar = Arg(user_convert="raw")
TEMP = Func("", Ret(),
            [
                ar.clone("BufferView<$T$>", "d_input"),
                ar.clone("BufferView<$T$>", "d_output"),
                ar.clone("int", "num_items"),
            ])
TEMP.set_template(lcub)

SubtractLeftCopy = TEMP.rename("SubtractLeftCopy")
SubtractRightCopy = TEMP.rename("SubtractRightCopy")

DeviceAdjacentDifference.add_funcs([SubtractLeftCopy.instantiate([("$T$", t)]) for t in value_types])
DeviceAdjacentDifference.add_funcs([SubtractRightCopy.instantiate([("$T$", t)]) for t in value_types])

TEMP = Func("", Ret(),
            [
                ar.clone("BufferView<$T$>", "d_input"),
                ar.clone("int", "num_items"),
            ])
TEMP.set_template(lcub)

SubtractLeft = TEMP.rename("SubtractLeft")
SubtractRight = TEMP.rename("SubtractRight")

DeviceAdjacentDifference.add_funcs([SubtractLeft.instantiate([("$T$", t)]) for t in value_types])
DeviceAdjacentDifference.add_funcs([SubtractRight.instantiate([("$T$", t)]) for t in value_types])

DeviceAdjacentDifference.write(folder="../")




