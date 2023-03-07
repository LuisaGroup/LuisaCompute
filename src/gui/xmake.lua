target("glfw")
_config_project({
	project_kind = "shared"
})
add_files("../ext/glfw/src/*.c")
add_includedirs("../ext/glfw/include", {
	public = true
})
add_defines("_GLFW_BUILD_DLL")
if is_plat("windows") then
	add_defines("_GLFW_WIN32")
end
add_syslinks("User32", "Gdi32", "Shell32")
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
add_deps("glfw", "lc-runtime")
if is_plat("windows") then
	add_defines("GLFW_EXPOSE_NATIVE_WIN32")
end
target_end()