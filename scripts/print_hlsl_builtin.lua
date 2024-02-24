-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
local files_list = {'accel_process', 'bindless_upload', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs',
                    'bc6_trymode_le10cs', 'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs',
                    'bc7_trymode_456cs', 'hlsl_header', 'raytracing_header', 'tex2d_bindless', 'tex3d_bindless',
                    'compute_quad', 'determinant', 'inverse', 'indirect', 'resource_size', 'accel_header', 'copy_sign',
                    'bindless_common', 'auto_diff', "reduce"}
local lib = import("lib")

local hlsl_builtin_path = path.join(os.projectdir(), "src/backends/common/hlsl/builtin")

function main()
    local sb = lib.StringBuilder()
    local ss = lib.StringBuilder()
    local arr_ss = lib.StringBuilder()
    local func_ss = lib.StringBuilder()
    arr_ss:add("#pragma once\n#include <luisa/core/stl/string.h>\nnamespace lc_hlsl{\n")
    func_ss:add([[
static luisa::string_view get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, luisa::string_view> dict;
        Dict(){
]])
    for i, file in ipairs(files_list) do
        ss:clear()
        sb:clear()
        local f = io.open(path.join(hlsl_builtin_path, file), "r")
        ss:add(f:read("*a"))
        f:close()
        sb:add('namespace lc_hlsl{\nchar '):add(file):add("[]={")
        local array_len = lib.to_byte_array(ss, sb)
        sb:add("};\n}")
        sb:write_to(path.join(hlsl_builtin_path, file .. ".cpp"))
        arr_ss:add('extern char '):add(file):add('[];\n')
        func_ss:add('\t\t\tdict.try_emplace("'):add(file):add('", luisa::string_view{'):add(file):add(', '):add(
            tostring(array_len)):add('});\n')
    end
    func_ss:add([[		}
	};
	static Dict dict;
	return dict.dict.find(ss)->second;
}
}]])
    ss:dispose()
    sb:dispose()
    arr_ss:add(func_ss)
    arr_ss:write_to(path.join(hlsl_builtin_path, "hlsl_builtin.hpp"))
    arr_ss:dispose()
    func_ss:dispose()
end
