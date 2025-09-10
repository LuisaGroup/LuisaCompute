if (get_config("vk_backend") or get_config("dx_backend")) and is_host("windows") then
    includes("hlsl")
end
if get_config("_lc_vk_sdk_dir") and (get_config("cuda_backend") or get_config("cpu_backend")) then
    target("lc-vulkan-swapchain")
    _config_project({
        project_kind = "object"
    })
    set_values("vk_public", true)
    add_headerfiles("vulkan_instance.h")
    add_defines("LUISA_USE_VOLK", {
        public = true
    })
    add_files("vulkan_swapchain.cpp", "vulkan_instance.cpp", "volk_build.c")
    add_deps("lc-core", "volk")
    if is_plat("linux") then
        add_syslinks("xcb", "X11", {
            public = true
        })
    end
    target_end()
end

if get_config("toy_c_backend") then
    target("lc-clanguage-codegen")
    _config_project({
        project_kind = "static"
    })
    add_deps("lc-core", "lc-ast", "lc-vstl")
    add_files("c_codegen/*.cpp", "hlsl/string_builder.cpp")
    set_pcxxheader("c_codegen/lc_ccodegen_pch.h")
    target_end()
end
