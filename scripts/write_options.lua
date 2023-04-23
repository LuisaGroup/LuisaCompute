local lib = import("lib")

local function find_llvm()
	local clang_name = "clang"
	local path_split = ":"
	if os.is_host("linux") and os.isfile("/usr/bin/llvm-ar") then
		return "/usr"
	elseif os.is_host("macosx") then
		import("lib.detect.find_tool")
		local bindir = find_path("llvm-ar", "/usr/local/Cellar/llvm/*/bin")
		if bindir then
			return path.directory(bindir)
		end
	else
		clang_name = "clang.exe"
		path_split = ";"
	end
	local path_str = os.getenv("PATH")
	if path_str then
		local paths = lib.string_split(path_str, path_split)
		for i, pth in ipairs(paths) do
			if os.is_host("windows") then
				pth = lib.string_replace(pth, "\\", "/")
			end
			if os.isfile(path.join(pth, clang_name)) then
				return path.directory(pth)
			end
		end
	end
	return nil
end
function main(...)
	local llvm_path = find_llvm()
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
	for i, v in ipairs(args) do
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
