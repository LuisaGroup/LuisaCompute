target("lc-backend-vk")
set_basename("luisa-backend-vk")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_headerfiles("*.h")
add_files("*.cpp")
set_pcxxheader("lc_vk_pch.h")

on_load(function(target)
    if target:is_plat("windows") then
        target:add("defines", "VK_USE_PLATFORM_WIN32_KHR")
    elseif target:is_plat("linux") then
        target:add("defines", "VK_USE_PLATFORM_XCB_KHR")
    end
    local function rela(p)
        return path.normalize(path.join(os.scriptdir(), p))
    end
    target:add("headerfiles", rela("../common/default_binary_io.h"))
    target:add("deps", "lc-volk")
    if target:is_plat("macosx") then
        target:add("files", rela("../common/moltenvk_surface.mm"))
        target:add("frameworks", "Foundation", "Metal", "QuartzCore", "AppKit")
    end
    if has_config("lc_vk_cuda_interop") then
        target:add("defines", "LUISA_VULKAN_ENABLE_CUDA_INTEROP")
        target:add("links", "nvrtc_static", "cudart_static", "cuda")
        target:add('deps', '_lc_cuda_base')
    end
end)
target_end()
