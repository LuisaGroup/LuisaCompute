target("lcapi")
local my_table = {
	project_kind = "shared",
	enable_exception = true
}
on_load(function(target)
	local python_str = "python"
	local py_path = get_config("py_path")
	local lib_path = path.join(py_path, "libs")
	local lib_ext
	if is_plat("windows") then
		lib_ext = ".lib"
	else
		lib_ext = ".a"
	end
	target:add("linkdirs", lib_path)
	for _, filepath in ipairs(os.files(path.join(lib_path, "*")))do
		local filename = path.filename(filepath)
		local basename = path.basename(filename)
		local ext = path.extension(filename)
		if ext == lib_ext and basename:lower():sub(1, python_str:len()) == python_str then
			target:add("links", basename)
		end		
	end
	target:add("includedirs", path.join(py_path, "include"))
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
