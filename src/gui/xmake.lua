target("lc-glfw")
_config_project({
	project_kind = "shared"
})
add_files("../ext/glfw/src/*.c")
add_includedirs("../ext/glfw/include", {
	public = true
})
add_defines("_GLFW_BUILD_DLL")
if is_plat("linux") then
	add_defines("_GLFW_X11", "_DEFAULT_SOURCE")
elseif is_plat("windows") then
	add_defines("_GLFW_WIN32")
	add_syslinks("User32", "Gdi32", "Shell32")
end
target_end()
-- _config_project({
-- 	project_name = "imgui",
-- 	project_kind = "shared"
-- })
-- add_includedirs("../ext/imgui/", {public = true})

-- add_files("../ext/imgui/imgui/*.cpp")
target("lc-gui")
_config_project({
	project_kind = "shared"
})
add_files("*.cpp")
add_defines("LC_GUI_EXPORT_DLL", "GLFW_DLL")
add_deps("lc-glfw", "lc-runtime")
if is_plat("windows") then
	add_defines("GLFW_EXPOSE_NATIVE_WIN32")
end
target_end()
