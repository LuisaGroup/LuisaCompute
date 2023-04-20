files_list = ['accel_process', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs', 'bc6_trymode_le10cs', 'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs', 'hlsl_header', 'raytracing_header']
special_map = {
    "\t": "\\t",
    "\r": "",
    "\n": "\\n",
    "\\": "\\\\",
    "\"":"\\\"",
    "\'":"\\'",
}
header = '#include "hlsl_config.h"\n'
for file in files_list:
    f = open(file, "r")
    ss = f.read()
    f.close()
    data = '#include "hlsl_config.h"\n'
    data += f"LC_HLSL_EXTERN int {file}_size={len(ss)};\n"
    data += f"LC_HLSL_EXTERN char {file}" + "[]={"
    for i in ss:
        d = special_map.get(i)
        if d != None:
            data += "'" + d + "',"
        else:
            data += "'" + i + "',"
    data = data[0:len(data)-1]
    data += "};\n"
    data += f"LC_HLSL_EXTERN char *get_{file}()" + "{return " + file + ";}\n"
    data += f"LC_HLSL_EXTERN int get_{file}_size()" + "{return " + file + "_size;}\n"
    header += f"LC_HLSL_EXTERN int get_{file}_size();\n"
    header += f"LC_HLSL_EXTERN char *get_{file}();\n"
    f= open(file + ".c", "w")
    f.write(data)
    f.close()
f = open("hlsl_builtin.h", "w")
f.write("#pragma once\n")
f.write(header)
f.close()