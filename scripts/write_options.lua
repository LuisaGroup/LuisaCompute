local lib = import("lib")
local function find_process_path(process)
	local cut
	local is_win = os.is_host("windows")
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

local function find_clangcl()
	return find_process_path("clang-cl.exe") ~= nil
end

local function find_llvm()
	local clang_name = "clang"
	if os.is_host("linux") and os.isfile("/usr/bin/llvm-ar") then
		return "/usr"
	elseif os.is_host("macosx") then
		import("lib.detect.find_path")
		local bindir = find_path("llvm-ar", "/usr/local/Cellar/llvm/*/bin")
		if bindir then
			return path.directory(bindir)
		end
	else
		clang_name = clang_name .. ".exe"
	end
	local clang_bin = find_process_path(clang_name)
	if clang_bin then
		return lib.string_replace(path.directory(clang_bin), "\\", "/")
	end
	return nil
end
function main(...)
	-- workaround xmake
	local args = {}
	for i, v in ipairs({...}) do
		local kv = lib.string_split(v, "=")
		if table.getn(kv) == 2 then
			args[kv[1]] = kv[2]
		end
	end
	local sb = lib.StringBuilder()
	sb:add("lc_toolchain = {\n")
	local toolchain = args["toolchain"]
	local sdk_path
	local is_win = os.is_host("windows")
	if toolchain then
		args["toolchain"] = nil
		if toolchain == "llvm" then
			if is_win then
				sdk_path = find_clangcl()
				toolchain = "clang-cl"
			else
				sdk_path = find_llvm()
			end
		end
	else
		-- llvm first
		if is_win then
			toolchain = "clang-cl"
			sdk_path = find_clangcl()
		else
			toolchain = "llvm"
			sdk_path = find_llvm()
		end

		if not sdk_path then
			if is_win then
				toolchain = "msvc"
			else
				toolchain = "gcc"
			end
		end
	end
	if os.is_host("macosx") then
		sb:add('\tmm = "clang",\n\tmxx = "clang++",\n')
	end
	sb:add("\ttoolchain = \""):add(toolchain):add("\",\n")
	if toolchain == "llvm" and sdk_path then
		sb:add("\tsdk = \""):add(sdk_path):add("\",\n")
	end
	sb:add("}\nfunction get_options()\n\treturn {\n")
	local py = args["python"] ~= nil
	if py then
		args["python"] = nil
	end
	if os.is_host("linux") and not args["enable_mimalloc"] then
		args["enable_mimalloc"] = "false"
	end
	sort_key(args, function(k, v)
		if not (v == "true" or v == "false") then
			v = '"' .. v .. '"'
		end
		sb:add("\t\t"):add(k .. " = " .. v):add(',\n')
	end)
	-- python

	if py and args["py_include"] == nil and os.is_host("windows") then
		local py_path = find_process_path("python.exe")
		if py_path then
			sb:add("\t\tpy_include = \""):add(lib.string_replace(path.join(py_path, "include"), "\\", "/")):add(
							"\",\n\t\tpy_linkdir = \"")
			local py_linkdir = path.join(py_path, "libs")
			sb:add(lib.string_replace(py_linkdir, "\\", "/"))
			local py = "python"
			sb:add("\",\n")
			local files = {}
			for _, filepath in ipairs(os.files(path.join(py_linkdir, "*.lib"))) do
				local lib_name = path.basename(filepath)
				if #lib_name >= #py and lib_name:sub(1, #py):lower() == py then
					table.insert(files, lib_name)
				end
			end
			if #files > 0 then
				sb:add("\t\tpy_libs = \"")
				for i, v in ipairs(files) do
					sb:add(v .. ";")
				end
				sb:add("\",\n")
			end
		end
	end
	sb:add("\t}\nend\n")
	sb:write_to(path.join(os.scriptdir(), "options.lua"))
	sb:dispose()
end
