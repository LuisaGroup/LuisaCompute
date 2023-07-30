# from wrapper import *
# from dcub_template import value_types, id_type
# import dcub_template
# import lcub_template

# website = "https://nvlabs.github.io/cub/structcub_1_1_device_select.html"
# class_name = "DeviceSelect"

# dcub = dcub_template.template(website)
# DeviceSelect = Class(class_name)
# DeviceSelect.set_template(dcub)

# Flagged = Func("Flagged", Ret(),
#                 [
#                     Arg("const $T$*", "d_in"),
#                     Arg("const $I$*", "d_flags"),
#                     Arg("$T$*", "d_out"),
#                     Arg("$I$*", "d_num_selected_out"),
#                     Arg("int", "num_items")
#                 ])
# Flagged.set_template(dcub)
# DeviceSelect.add_funcs([ Flagged.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# # UniqueByKey = Func("UniqueByKey", Ret(),
# #                 [
# #                     Arg("const $I$*", "d_keys_in"),
# #                     Arg("const $T$*", "d_values_in"),
# #                     Arg("$I$*", "d_keys_out"),
# #                     Arg("$T$*", "d_values_out"),
# #                     Arg("$I$*", "d_num_selected_out"),
# #                     Arg("int", "num_items")
# #                 ])
# # UniqueByKey.set_template(dcub)
# # DeviceSelect.add_funcs([ UniqueByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# Unique = Func("Unique", Ret(),
#                 [ 
#                     Arg("const $I$*", "d_in"),
#                     Arg("$I$*", "d_out"),
#                     Arg("$I$*", "d_num_selected_out"),
#                     Arg("int", "num_items")
#                 ])
# Unique.set_template(dcub)
# DeviceSelect.add_funcs([ Unique.instantiate([("$I$", id_type)])])

# DeviceSelect.write(src_ext=".cu", folder="../private/dcub/")


# #LCUB
# lcub = lcub_template.template(website)
# DeviceSelect = Class(class_name)
# DeviceSelect.set_template(lcub)


# ar = Arg(user_convert="raw")

# Flagged = Func("Flagged", Ret(),
#                 [
#                     ar.clone("BufferView<$T$>", "d_in"),
#                     ar.clone("BufferView<$I$>", "d_flags"),
#                     ar.clone("BufferView<$T$>", "d_out"),
#                     ar.clone("BufferView<$I$>", "d_num_selected_out"),
#                     ar.clone("int", "num_items")
#                 ])
# Flagged.set_template(lcub)
# DeviceSelect.add_funcs([ Flagged.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# # UniqueByKey = Func("UniqueByKey", Ret(),
# #                 [
# #                     ar.clone("BufferView<$I$>", "d_keys_in"),
# #                     ar.clone("BufferView<$T$>", "d_values_in"),
# #                     ar.clone("BufferView<$I$>", "d_keys_out"),
# #                     ar.clone("BufferView<$T$>", "d_values_out"),
# #                     ar.clone("BufferView<$I$>", "d_num_selected_out"),
# #                     ar.clone("int", "num_items")
# #                 ])
# # UniqueByKey.set_template(lcub)
# # DeviceSelect.add_funcs([ UniqueByKey.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

# Unique = Func("Unique", Ret(),
#                 [ 
#                     ar.clone("BufferView<$I$>", "d_in"),
#                     ar.clone("BufferView<$I$>", "d_out"),
#                     ar.clone("BufferView<$I$>", "d_num_selected_out"),
#                     ar.clone("int", "num_items")
#                 ])
# Unique.set_template(lcub)
# DeviceSelect.add_funcs([ Unique.instantiate([("$I$", id_type)])])

# DeviceSelect.write(folder="./")


from wrapper import *
from dcub_template import value_types, id_type
import dcub_template
import lcub_template

website = "https://nvlabs.github.io/cub/structcub_1_1_device_partition.html"
class_name = "DevicePartition"

dcub = dcub_template.template(website)
DevicePartition = Class(class_name)
DevicePartition.set_template(dcub)

Flagged = Func("Flagged", Ret(),
                [
                    Arg("const $T$*", "d_in"),
                    Arg("const $I$*", "d_flags"),
                    Arg("$T$*", "d_out"),
                    Arg("$I$*", "d_num_selected_out"),
                    Arg("int", "num_items")
                ])
Flagged.set_template(dcub)
DevicePartition.add_funcs([ Flagged.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

DevicePartition.write(src_ext=".cu", folder="../private/dcub/")

#LCUB

lcub = lcub_template.template(website)
DevicePartition = Class(class_name)
DevicePartition.set_template(lcub)

ar = Arg(user_convert="raw")

Flagged = Func("Flagged", Ret(),
                [
                    ar.clone("BufferView<$T$>", "d_in"),
                    ar.clone("BufferView<$I$>", "d_flags"),
                    ar.clone("BufferView<$T$>", "d_out"),
                    ar.clone("BufferView<$I$>", "d_num_selected_out"),
                    ar.clone("int", "num_items")
                ])
Flagged.set_template(lcub)
DevicePartition.add_funcs([ Flagged.instantiate([("$T$", t), ("$I$", id_type)]) for t in value_types])

DevicePartition.write(folder="../")
                    