-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
local files_list = {'accel_process', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs', 'bc6_trymode_le10cs',
                    'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs',
                    'hlsl_header', 'raytracing_header'}
local special_map = {}
local lib = import("lib")
local function insert(map, k, v)
	map[k] = v
end
insert(special_map, lib.char("\t"), "\\t")
insert(special_map, lib.char("\r"), "")
insert(special_map, lib.char("\n"), "\\n")
insert(special_map, lib.char("\\"), "\\\\")
insert(special_map, lib.char("\""), "\\\"")
insert(special_map, lib.char("\'"), "\\'")
local header = lib.StringBuilder('#pragma once\n#include "hlsl_config.h"\n')
local hlsl_builtin_path = path.join(os.projectdir(), "src/backends/common/hlsl/builtin")

function main()
	local char_begin = lib.char("'")
	local cut = lib.char(',')
	for i, file in ipairs(files_list) do
		local f = io.open(path.join(hlsl_builtin_path, file), "r")
		local ss = lib.StringBuilder(f:read("*a"))
		f:close()
		local sb = lib.StringBuilder()
		sb:add('#include "hlsl_config.h"\nLC_HLSL_EXTERN int ')
		:add(file)
		:add("_size=")
		:add(tostring(ss:size()))
		:add(";\nLC_HLSL_EXTERN char ")
		:add(file)
		:add("[]={")
		for idx = 1, ss:size() do
			local i = ss:get(idx)
			local d = special_map[i]
			sb:add_char(char_begin)
			if d then
				sb:add(d)
			else
				sb:add_char(i)
			end
			sb:add("',")
		end
		ss:dispose()
		sb:erase(1)
		sb:add("};\nLC_HLSL_EXTERN char *get_"):add(file):add("(){return "):add(file):add(";}\nLC_HLSL_EXTERN int get_"):add(
						file):add("_size(){return "):add(file):add("_size;}\n")
		header:add("LC_HLSL_EXTERN int get_"):add(file):add("_size();\n"):add("LC_HLSL_EXTERN char *get_"):add(file):add(
						"();\n")
		sb:write_to(path.join(hlsl_builtin_path, file .. ".c"))
		sb:dispose()
	end
	header:write_to(path.join(hlsl_builtin_path, "hlsl_builtin.h"))
	header:dispose()
end
