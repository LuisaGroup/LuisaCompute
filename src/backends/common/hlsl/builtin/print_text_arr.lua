-- use command:
-- xmake lua printer_text_arr.lua
-- to execute this script and gen new files
local files_list = {'accel_process', 'bc6_encode_block', 'bc6_header', 'bc6_trymode_g10cs', 'bc6_trymode_le10cs',
                    'bc7_encode_block', 'bc7_header', 'bc7_trymode_02cs', 'bc7_trymode_137cs', 'bc7_trymode_456cs',
                    'hlsl_header', 'raytracing_header'}
local special_map = {}
local function insert(map, k, v)
	map[k] = v
end
insert(special_map, "\t", "\\t")
insert(special_map, "\r", "")
insert(special_map, "\n", "\\n")
insert(special_map, "\\", "\\\\")
insert(special_map, "\"", "\\\"")
insert(special_map, "\'", "\\'")
local header = '#include "hlsl_config.h"\n'

function main()
	for i, file in ipairs(files_list) do
		local f = io.open(path.join(os.scriptdir(), file), "r")
		local ss = f:read("*a")
		f:close()
        f = io.open(path.join(os.scriptdir(), file .. ".c"), "w")
		f:write('#include "hlsl_config.h"\nLC_HLSL_EXTERN int ')
		f:write(file)
		f:write("_size=")
		f:write(#ss)
		f:write(";\nLC_HLSL_EXTERN char ")
		f:write(file)
		f:write("[]={")
		for idx = 1, #ss do
			local i = ss:sub(idx, idx)
			local d = special_map[i]
			if d ~= nil then
				f:write("'" .. d .. "'")
			else
				f:write("'" .. i .. "'")
			end
			if idx ~= #ss then
				f:write(",")
			end
		end
		f:write("};\nLC_HLSL_EXTERN char *get_")
		f:write(file)
		f:write("(){return ")
		f:write(file)
		f:write(";}\nLC_HLSL_EXTERN int get_")
		f:write(file)
		f:write("_size(){return ")
		f:write(file)
		f:write("_size;}\n")
        f:close()
        header = header .. "LC_HLSL_EXTERN int get_" .. file .. "_size();\n" .. "LC_HLSL_EXTERN char *get_" .. file .. "();\n"
	end
    local f = io.open(path.join(os.scriptdir(), "hlsl_builtin.h"), "w")
    f:write("#pragma once\n")
    f:write(header)
    f:close()
end
