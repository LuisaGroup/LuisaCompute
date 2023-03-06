_config_project({
	project_name = "lcapi",
	project_kind = "shared",
	enable_exception = true
})
on_load(function(target)
	local version_table = {}
	for str in string.gmatch(get_config("py_version"), "([^_]+)") do
		table.insert(version_table, str)
	end
	local legal_version = (table.getn(version_table) == 2) and version_table[1] == '3'
	if legal_version then
		local num = tonumber(version_table[2])
		if num == nil then
			legal_version = false
		end
	end
	if legal_version then
		local py_name = "python" .. version_table[1] .. version_table[2]
		local py_path = get_config("py_path")
		target:add("links", "python3", py_name)
		target:add("linkdirs", py_path .. "/libs")
		target:add("includedirs", "src/ext/pybind11/include", py_path .. "/include", "src/ext/stb/")
		target:add("deps", "lc-runtime", "lc-gui")
		target:add("defines", "LC_AST_EXCEPTION")
	else
		target:set("enabled", false)
		utils.error("Illegal python version argument. please use argument like 3_9(for python 3.9) or 3_10(for python 3.10)")
		return
	end
end)
add_files("*.cpp")

after_build(function(target)
	local bdPath = target:targetdir()
	if is_plat("windows") then
		os.cp(bdPath .. "/lcapi.dll", bdPath .. "/lcapi.pyd")
	end
end)