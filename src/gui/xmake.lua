target("glfw")
set_basename("lc-ext-glfw")
_config_project({
    project_kind = "static"
})
add_headerfiles("../ext/glfw/include/**.h")
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
elseif is_plat("macosx") then
    add_files("../ext/glfw/src/*.m")
    add_mflags("-fno-objc-arc")
    add_defines("_GLFW_COCOA")
    add_frameworks("Foundation", "Cocoa", "IOKit", "OpenGL", "QuartzCore")
end
target_end()


target("imgui")
set_basename("lc-ext-imgui")
_config_project({
    project_kind = "shared"
})
add_headerfiles("../ext/imgui/*.h", "../ext/imgui/backends/*.h")
add_files("../ext/imgui/*.cpp", "../ext/imgui/backends/imgui_impl_glfw.cpp")
add_includedirs("../ext/imgui", "../ext/imgui/backends", {
    public = true
})
if is_plat("windows") then
    add_defines("IMGUI_API=__declspec(dllexport)")
end
add_defines("ImDrawIdx=unsigned int", {public = true})
add_deps("glfw", "lc-dsl")
target_end()

target("lc-gui")
_config_project({
    project_kind = "shared"
})
add_headerfiles("../../include/luisa/gui/**.h")
add_files("*.cpp")
add_defines("LC_GUI_EXPORT_DLL", "GLFW_DLL")
add_deps("glfw", "lc-runtime", "imgui")
if is_plat("windows") then
    add_defines("IMGUI_API=__declspec(dllimport)", {public = true})
end
target_end()
