target("lc-backend-vk")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl")
set_values("vk_path", LCVulkanPath)
add_rules("lc_vulkan")
if is_plat("windows") then
    add_deps("lc-copy-dxc")
end
add_files("**.cpp")
-- TODO: use dxc for vulkan, only windows temporarily
if is_plat("windows") then
    add_defines("VK_USE_PLATFORM_WIN32_KHR")
elseif is_plat("linux") then
    add_defines("VK_USE_PLATFORM_DIRECTFB_EXT")
end
target_end()
