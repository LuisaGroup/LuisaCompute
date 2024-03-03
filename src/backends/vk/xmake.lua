target("lc-backend-vk")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-builtin")
add_packages("vulkansdk")
add_headerfiles("*.h", "../common/default_binary_io.h", "../common/hlsl/*.h")
add_files("*.cpp", "../common/default_binary_io.cpp", "../common/hlsl/*.cpp")
set_pcxxheader("pch.h")
-- TODO: use dxc for vulkan, only windows temporarily
if is_plat("windows") then
    add_defines("VK_USE_PLATFORM_WIN32_KHR")
elseif is_plat("linux") then
    add_defines("VK_USE_PLATFORM_XCB_KHR")
end
add_defines("USE_SPIRV")
target_end()
