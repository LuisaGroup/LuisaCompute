target("lcapi")
on_load(function(target)
	local function split_str(str, chr, func)
		for part in string.gmatch(str, "([^" .. chr .. "]+)") do
			func(part)
		end
	end
	local py_include = get_config("py_include")
	split_str(py_include, ';', function(v)
		target:add("includedirs", v)
	end)
	local py_linkdir = get_config("py_linkdir")
	local py_libs = get_config("py_libs")
	if type(py_linkdir) == "string" then
		split_str(py_linkdir, ';', function(v)
			target:add("linkdirs", v)
		end)
	end
	if type(py_libs) == "string" then
		split_str(py_libs, ';', function(v)
			target:add("links", v)
		end)
	end
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("includedirs", rela("../ext/stb/"), rela("../ext/pybind11/include"))
end)

_config_project({
	project_kind = "shared",
	enable_exception = true
})
set_pcxxheader("pch.h")
add_cxflags("/bigobj", {
	tools = "cl"
})
add_headerfiles("*.h")
add_files("*.cpp")
add_deps("lc-runtime", "lc-gui")
after_build(function(target)
	if is_plat("windows") then
		local bdPath = target:targetdir()
		os.cp(path.join(bdPath, "lcapi.dll"), path.join(bdPath, "lcapi.pyd"))
	end
end)
target_end()
