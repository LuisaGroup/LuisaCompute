local version_table = {}
local split_size = 0
for str in string.gmatch(PythonVersion, "([^_]+)") do
	table.insert(version_table, str)
	split_size = split_size + 1
end
local legal_version = (split_size == 2) and version_table[1] == '3'
if legal_version then
	local num = tonumber(version_table[2])
	if num == nil then
		legal_version = false
	end
end
if legal_version then
	_config_project({
		project_name = "lcapi",
		project_kind = "shared",
		enable_exception = true
	})
	local pyName = "python" .. version_table[1] .. version_table[2]
	add_links("python3", pyName)
	add_linkdirs(PythonPath .. "/libs")
	add_includedirs("../ext/pybind11/include", PythonPath .. "/include", "../ext/stb/")
	add_files("*.cpp")
	add_deps("lc-runtime")
	add_defines("LC_AST_EXCEPTION")
	add_deps("lc-gui")
	after_build(function(target)
		local bdPath = target:targetdir()
		if is_plat("windows") then
			os.cp(bdPath .. "/lcapi.dll", bdPath .. "/lcapi.pyd")
		end
	end)
else
	target("_lc_illegal_py")
	set_kind("phony")
	on_load(function(target)
		utils.error("Illegal python version argument. please use argument like 3_9(for python 3.9) or 3_10(for python 3.10)")
	end)
	target_end()
end
