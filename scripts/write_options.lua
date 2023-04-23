local lib = import("lib")

local function find_llvm(envs)
	if os.is_host("linux") and os.isfile("/usr/bin/llvm-ar") then
		return "/usr"
	elseif os.is_host("macosx") then
		import("lib.detect.find_tool")
		local bindir = find_path("llvm-ar", "/usr/local/Cellar/llvm/*/bin")
		if bindir then
			return path.directory(bindir)
		end
	end

	local path_str = envs["PATH"]
	if path_str then
		local paths = lib.string_split(path_str, ";")
		for i, path in ipairs(paths) do
			if os.is_host("windows") then
				path = lib.string_replace(path, "\\", "/")
			end
			local path_parts = lib.string_split(path, "/")
			local is_llvm = false
			local result = ""
			for i, v in ipairs(path_parts) do
				result = result .. v .. "/"
				if lib.string_contains(v:lower(), "llvm") then
					is_llvm = true
					break
				end
			end
			if is_llvm then
				return result
			end
		end
	end
	local llvm_dirs = {
		LLVM_SDK = true,
		LLVM_DIR = true
	}
	for k, v in pairs(envs) do
		if llvm_dirs[k] then
			return v
		end
	end

end
function main(...)
	local envs = os.getenvs()
	local llvm_path = find_llvm(envs)
	local option_file = io.open(path.join(os.projectdir(), "scripts/options.lua"), "w")
	option_file:write("lc_config = {\n")
	if llvm_path then
		option_file:write("\ttoolchain = \"llvm\",\n\tsdk = \"" .. llvm_path .. "\"\n}\n")
	elseif os.is_host("linux") then
		option_file:write("\ttoolchain = \"gcc\"\n}\n")
	elseif os.is_host("windows") then
		option_file:write("\ttoolchain = \"msvc\"\n}\n")
	else
		option_file:write("}\n")
	end
	option_file:write("function get_options()\n\treturn {\n")
	local args = {...}
    for i,v in ipairs(args) do
        local kv = lib.string_split(v, "=")
        if table.getn(kv) == 2 then
            local v = kv[2]
            if not (v == "true" or v == "false") then
                v = '"' .. v .. '"'
            end
            option_file:write("\t\t" .. kv[1] .. " = " .. v .. ',\n')
        end
    end

	option_file:write("\t}\nend\n")
	option_file:close()
end
