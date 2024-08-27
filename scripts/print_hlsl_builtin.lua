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
    local test_zip_dir = path.join(os.projectdir(), "bin/release/test_zip.exe")
    local sb = lib.StringBuilder()
    local ss = lib.StringBuilder()
    local arr_ss = lib.StringBuilder()
    local func_ss = lib.StringBuilder()
    arr_ss:add("#pragma once\n#include <luisa/core/stl/string.h>\nnamespace lc_hlsl{\n")
    func_ss:add([[
struct HLSLCompressedHeader {
    void const* ptr;
    size_t compressed_size;
    size_t uncompressed_size;
};
static HLSLCompressedHeader get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, HLSLCompressedHeader> dict;
        Dict(){
]])
    for i, file in ipairs(files_list) do
        -- make this file ignored by git
        local compressed_file = file .. ".msi"
        os.runv(test_zip_dir, {path.join(hlsl_builtin_path, file),path.join(hlsl_builtin_path,  compressed_file)})
        ss:clear()
        sb:clear()
        local uncompressed_size
        try{function()
            local ff = io.open(path.join(hlsl_builtin_path, file), "rb")
            uncompressed_size = ff:size()
        end}
        local f = io.open(path.join(hlsl_builtin_path, compressed_file), "rb")
        ss:add(f:read("*a"))
        f:close()
        sb:add('namespace lc_hlsl{\nunsigned char '):add(file):add("["):add(tostring(math.tointeger(ss:size()))):add("]={")
        local array_len = tostring(math.tointeger(lib.to_byte_array(ss, sb)))
        sb:add("};\n}")
        sb:write_to(path.join(hlsl_builtin_path, file .. ".cpp"))
        arr_ss:add('extern unsigned char '):add(file):add('[];\n')
        func_ss:add('\t\t\tdict.try_emplace("'):add(file):add('", HLSLCompressedHeader{'):add(file):add(', '):add(array_len):add(', '):add(tostring(math.tointeger(uncompressed_size))):add('});\n')
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
