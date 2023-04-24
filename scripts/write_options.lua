local lib = import("lib")
local my_is_host
local function find_process_path(process)
	local cut
	local is_win = my_is_host("windows")
	if is_win then
		cut = ";"
	else
		cut = ":"
	end
	local path_str = os.getenv("PATH")
	if path_str then
		local paths = lib.string_split(path_str, cut)
		for i, pth in ipairs(paths) do
			if os.isfile(path.join(pth, process)) then
				return pth
			end
		end
	end
	return nil
end

local function sort_key(map, func)
	local keys = {}
	for k, v in pairs(map) do
		table.insert(keys, k)
	end
	table.sort(keys)
	for i, v in ipairs(keys) do
		func(v, map[v])
	end
end

local function find_llvm()
	local clang_name = "clang"
	if my_is_host("linux") and os.isfile("/usr/bin/llvm-ar") then
		return "/usr"
	elseif my_is_host("macosx") then
		import("lib.detect.find_tool")
		local bindir = find_path("llvm-ar", "/usr/local/Cellar/llvm/*/bin")
		if bindir then
			return path.directory(bindir)
		end
	else
		clang_name = "clang.exe"
	end
	local clang_bin = find_process_path(clang_name)
	if clang_bin then
		return lib.string_replace(path.directory(clang_bin), "\\", "/")
	end
	return nil
end
function main(...)
	-- workaround xmake
	if type(os.is_host) == "function" then
		my_is_host = os.is_host
	elseif type(os.host) == "function" then
		my_is_host = function(p)
			return os.host() == p
		end
	else
		my_is_host = is_host
	end
	local args = {}
	for i, v in ipairs({...}) do
		local kv = lib.string_split(v, "=")
		if table.getn(kv) == 2 then
			args[kv[1]] = kv[2]
		end
	end
	local option_file = io.open(path.join(os.scriptdir(), "options.lua"), "w")
	option_file:write("lc_toolchain = {\n")
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
		elseif my_is_host("windows") then
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
	option_file:write("}\nfunction get_options()\n\treturn {\n")
	local py = args["python"] ~= nil
	if py then
		args["python"] = nil
	end
	sort_key(args, function(k, v)
		if not (v == "true" or v == "false") then
			v = '"' .. v .. '"'
		end
		option_file:write("\t\t")
		option_file:write(k .. " = " .. v)
		option_file:write(',\n')
	end)
	-- python

	if py and args["py_include"] == nil and my_is_host("windows") then
		local py_path = find_process_path("python.exe")
		if py_path then
			option_file:write("\t\tpy_include = \"")
			option_file:write(lib.string_replace(path.join(py_path, "include"), "\\", "/"))
			option_file:write("\",\n\t\tpy_linkdir = \"")
			local py_linkdir = path.join(py_path, "libs")
			option_file:write(lib.string_replace(py_linkdir, "\\", "/"))
			local py = "python"
			option_file:write("\",\n")
			local files = {}
			for _, filepath in ipairs(os.files(path.join(py_linkdir, "*.lib"))) do
				local lib_name = path.basename(filepath)
				if #lib_name >= #py and lib_name:sub(1, #py):lower() == py then
					table.insert(files, lib_name)
				end
			end
			if #files > 0 then
				option_file:write("\t\tpy_libs = \"")
				for i, v in ipairs(files) do
					option_file:write(v .. ";")
				end
				option_file:write("\",\n")
			end
		end
	end
	option_file:write("\t}\nend\n")
	option_file:close()
end
