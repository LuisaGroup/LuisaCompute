if get_config("vk_backend") or get_config("dx_backend") then
    includes("hlsl/builtin")
end
if (get_config("cuda_backend") or get_config("cpu_backend")) then
    target("lc-vulkan-swapchain")
    _config_project({
        project_kind = "static"
    })
    set_values("vk_public", true)
    add_headerfiles("vulkan_instance.h")
    add_defines("LUISA_USE_VOLK", {
        public = true
    })
    add_files("vulkan_swapchain.cpp", "vulkan_instance.cpp", "volk_build.c")
    add_deps("lc-core", "volk")
    if is_plat("linux") then
        add_syslinks("xcb", "X11", {public = true})
    end
    target_end()
end
