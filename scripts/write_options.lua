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
	local args = {}
	for i, v in ipairs({...}) do
		local kv = lib.string_split(v, "=")
		if table.getn(kv) == 2 then
			args[kv[1]] = kv[2]
		end
	end
	local option_file = io.open(path.join(os.projectdir(), "scripts/options.lua"), "w")
	option_file:write("lc_config = {\n")
	local toolchain = args["toolchain"]
	local sdk_path
	if toolchain then
		args["toolchain"] = nil
		if toolchain == "llvm" then
			sdk_path = find_llvm()
		end
	else
		sdk_path = find_llvm()
		if sdk_path then
			toolchain = "llvm"
		elseif os.is_host("windows") then
			toolchain = "msvc"
		else
			toolchain = "gcc"
		end
	end
	option_file:write("\ttoolchain = \"")
	option_file:write(toolchain)
	option_file:write("\",\n")
	if sdk_path then
		option_file:write("\tsdk = \"")
		option_file:write(sdk_path)
		option_file:write("\",\n")
	end
	option_file:write("}\n")
	option_file:write("function get_options()\n\treturn {\n")
	for k, v in pairs(args) do
		if not (v == "true" or v == "false") then
			v = '"' .. v .. '"'
		end
		option_file:write("\t\t")
		option_file:write(k)
		option_file:write(" = ")
		option_file:write(v)
		option_file:write(',\n')
	end

	option_file:write("\t}\nend\n")
	option_file:close()
end
