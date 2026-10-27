target("lcapi")
on_load(function(target)
	local lc_py_include = get_config("lc_py_include")
    if lc_py_include then
		target:add("includedirs", lc_py_include:split(";"))
	end
	local lc_py_linkdir = get_config("lc_py_linkdir")
	local lc_py_libs = get_config("lc_py_libs")
	if type(lc_py_linkdir) == "string" then
        target:add("linkdirs", lc_py_linkdir:split(";"))
	end
	if type(lc_py_libs) == "string" then
        target:add("links", lc_py_libs:split(";"))
	end
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("includedirs", rela("../ext/pybind11/include"))
end)

_config_project({
	project_kind = "shared",
	enable_exception = true
})
lc_set_pcxxheader("lcpy_pch.h")
add_cxflags("/bigobj", {tools = "cl"})
add_headerfiles("*.h")
add_files("*.cpp")
add_deps("lc-runtime", "lc-gui", "stb-image")
set_extension(".pyd")
target_end()
