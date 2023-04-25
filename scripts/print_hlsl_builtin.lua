-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
local files_list = {'accel_process', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs', 'bc6_trymode_le10cs',
                    'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs',
                    'hlsl_header', 'raytracing_header'}
local lib = import("lib")

local hlsl_builtin_path = path.join(os.projectdir(), "src/backends/common/hlsl/builtin")

function main()
	local sb = lib.StringBuilder()
	local ss = lib.StringBuilder()
	for i, file in ipairs(files_list) do
		ss:clear()
		sb:clear()
		local cut = lib.char(',')
		local f = io.open(path.join(hlsl_builtin_path, file), "r")
		ss:add(f:read("*a"))
		f:close()
		sb:add('#include "hlsl_config.h"\nLC_HLSL_EXTERN int '):add(file):add("_size="):add(tostring(ss:size())):add(
						";\nLC_HLSL_EXTERN char "):add(file):add("[]={")
		for idx = 1, ss:size() do
			local i = ss:get(idx)
			sb:add(tostring(i)):add_char(cut)
		end
		sb:erase(1)
		sb:add("};\nLC_HLSL_EXTERN char *get_"):add(file):add("(){return "):add(file):add(";}\nLC_HLSL_EXTERN int get_"):add(
						file):add("_size(){return "):add(file):add("_size;}\n")
		sb:write_to(path.join(hlsl_builtin_path, file .. ".c"))
	end
	ss:dispose()
	sb:dispose()
end
