-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
-- 'accel_process', 'accel_process_vk', 'bindless_upload', 'bindless_upload_vk', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs','bc6_trymode_le10cs', 'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs'
local files_list = {'hlsl_header', 'dx_linalg', 'hlsl_header_fallback', 'raytracing_header', 'tex2d_bindless',
                    'tex3d_bindless', 'compute_quad', 'determinant', 'inverse', 'indirect', 'resource_size',
                    'accel_header', 'copy_sign', 'bindless_common', 'auto_diff', "reduce", 'accel_process_vk.dxil',
                    'load_bdls.dxil', 'load_bdls_vk.dxil', 'set_accel4.dxil', 'bc6_encodeblock.dxil',
                    'bc6_trymodeg10.dxil', 'bc6_trymodele10.dxil', 'bc7_encodeblock.dxil', 'bc7_trymode02.dxil',
                    'bc7_trymode137.dxil', 'bc7_trymode456.dxil'}

local lib = import("lib")

for i,v in ipairs(files_list) do
    print("LC_HLSL_DECL_VARNAME(" .. lib.string_replace(v, ".dxil", "_dxil") .. ")")
end
print()
for i,v in ipairs(files_list) do
    print("LC_HLSL_INSERT_VARNAME(" .. lib.string_replace(v, ".dxil", "_dxil") .. ", \"" .. v .. "\")")
end