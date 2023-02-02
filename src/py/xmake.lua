function CompilePython(version)
	local versionName = "3" .. tostring(version)
	local pyName = "python" .. versionName
	local projectName = "lcapi" .. versionName
	_config_project({
		project_name = projectName,
		project_kind = "shared",
		enable_exception = true
	})
	add_links("python3", pyName)
	add_linkdirs("../ext/" .. pyName .. "/libs")
	add_includedirs("../ext/pybind11/include", "../ext/" .. pyName .. "/include", "../ext/stb/")
	add_files("*.cpp")
	add_deps("lc-runtime")
	add_defines("LC_AST_EXCEPTION")
	set_values("projectName", projectName)
	if EnableGUI then
		add_deps("lc-gui")
		add_defines("LC_PY_ENABLE_GUI")
	end
	after_build(function(target)
		local projectName = target:values("projectName")
		local bdPath = target:targetdir()
		if is_plat("windows") then
			os.cp(bdPath .. "/" .. projectName .. ".dll", bdPath .. "/lcapi.pyd")
		end
	end)
end
CompilePython(11)
--[[
for version = 1, 11, 1 do
	CompilePython(version)
end
]]