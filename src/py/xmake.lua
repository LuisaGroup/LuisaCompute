target("lcapi")
local my_table = {
	project_kind = "shared",
	enable_exception = true
}
on_load(function(target)
	local py_version = get_config("py_version")
	local py_path = get_config("py_path")
	local version_table = {}
	local function read_input()
		if type(py_version) ~= "string" or string.len(py_version) == 0 then
			return nil
		end
		for str in string.gmatch(py_version, "([^.]+)") do
			table.insert(version_table, str)
		end
		if (table.getn(version_table) ~= 2) or version_table[1] ~= "3" then
			return nil
		end
		local num = tonumber(version_table[2])
		if num == nil then
			return nil
		end
		return true
	end

	if read_input() then
		local py_name = "python" .. version_table[1] .. version_table[2]
		target:add("linkdirs", path.join(py_path, "libs"))
		target:add("links", "python3", py_name)
		target:add("includedirs", path.join(py_path, "include"))
	else
		target:set("enabled", false)
		utils.error("Illegal python version argument. please use argument like 3.9 (for python 3.9) or 3.10 (for python 3.10)")
		return
	end
end)

_config_project(my_table)
add_files("*.cpp")
add_includedirs("../ext/stb/", "../ext/pybind11/include")
add_deps("lc-runtime", "lc-gui")
after_build(function(target)
	local bdPath = target:targetdir()
	if is_plat("windows") then
		os.cp(path.join(bdPath, "lcapi.dll"), path.join(bdPath, "lcapi.pyd"))
	end
end)
target_end()
