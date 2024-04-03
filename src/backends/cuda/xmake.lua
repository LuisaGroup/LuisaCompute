if get_config("cuda_ext_lcub") then
    includes("lcub")
end
target("lc-cuda-base")
set_kind("phony")
on_load(function(target)
    import("detect.sdks.find_cuda")
	import("cuda_sdkdir")
    local cuda = find_cuda(cuda_sdkdir(nil, find_file))
    if cuda then
        local function set(key, value)
            if type(value) == "string" then
                target:add(key, value, {
                    public = true
                })
            elseif type(value) == "table" then
                for i, v in ipairs(value) do
                    target:add(key, v, {
                        public = true
                    })
                end
            end
        end
        set("linkdirs", cuda["linkdirs"])
        set("includedirs", cuda["includedirs"])
    else
        target:set("enabled", false)
        return
    end
    if is_plat("windows") then
        target:add("defines", "UNICODE", "_CRT_SECURE_NO_WARNINGS", {
            public = true
        })
        target:add("syslinks", "Cfgmgr32", "Advapi32", {
            public = true
        })
    end
end)
target_end()

target("lc-backend-cuda")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_defines("LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
add_deps("lc-vulkan-swapchain", "lc-runtime", "volk", "lc-cuda-base", "reproc")
if get_config("enable_ir") then
    add_deps("lc-ir")
end

if get_config("cuda_ext_lcub") then
    add_deps("luisa-compute-cuda-ext-lcub")
end

set_pcxxheader("pch.h")
add_headerfiles("*.h", "../common/default_binary_io.h")
add_files("*.cpp|cuda_texture_compression.cpp") -- TODO: support NVTT with XMake
add_files("extensions/cuda_denoiser.cpp", "extensions/cuda_dstorage.cpp", "extensions/cuda_pinned_memory.cpp")
add_links("cuda", "nvrtc")

after_build(function(target)
    import("lib.detect.find_file")
    import("detect.sdks.find_cuda")
	import("cuda_sdkdir")
    local cuda = find_cuda(cuda_sdkdir(nil, find_file))
    if cuda then
        local linkdirs = cuda["linkdirs"]
        local bin_dir = target:targetdir()
        if is_plat("windows") then
            for i, v in ipairs(linkdirs) do
                os.cp(path.join(v, "cudadevrt.lib"), bin_dir)
            end
        end
        -- TODO: linux
    end
end)
target_end()

target("lc-nvrtc")
_config_project({
    project_kind = "binary"
})
set_basename("luisa_nvrtc")
add_deps("lc-cuda-base")
add_links("nvrtc")
add_files("cuda_nvrtc_compiler.cpp")
target_end()
