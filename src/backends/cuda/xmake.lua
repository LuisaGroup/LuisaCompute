if get_config("cuda_ext_lcub") then
    includes("lcub")
end
target("lc-cuda-base")
set_kind("phony")
on_load(function(target)
    import("detect.sdks.find_cuda")
    import("cuda_sdkdir")
    local cuda = find_cuda(cuda_sdkdir())
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
    if get_config("lc_backend_lto") then
        target:set("policy", "build.optimization.lto", true)
        if get_config("lc_toolchain") == "llvm" then
            target:add("ldflags", "-fuse-ld=lld-link")
            target:add("shflags", "-fuse-ld=lld-link")
        end
    end
end)
target_end()

target("lc-backend-cuda")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-runtime", "lc-cuda-base", "reproc")
if get_config("enable_ir") then
    add_deps("lc-ir")
end

if get_config("cuda_ext_lcub") then
    add_deps("luisa-compute-cuda-ext-lcub")
end

set_pcxxheader("lc_cuda_pch.h")
add_headerfiles("*.h", "../common/default_binary_io.h")
on_load(function(target)
    local src_path = os.scriptdir()
    for _, filepath in ipairs(os.files(path.join(src_path, "*.cpp"))) do
        local file_name = path.filename(filepath)
        if file_name ~= "cuda_nvrtc_compiler.cpp" then
            target:add("files", filepath)
        end
    end
    if get_config("_lc_vk_sdk_dir") then
        target:add("defines", "LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
        target:add("deps", "lc-vulkan-swapchain", "volk")
    end
end)
add_files("extensions/cuda_denoiser.cpp", "extensions/cuda_dstorage.cpp", "extensions/cuda_pinned_memory.cpp")
add_links("cuda")

-- after_build(function(target)
--     import("lib.detect.find_file")
--     import("detect.sdks.find_cuda")
--     import("cuda_sdkdir")
--     local cuda = find_cuda(cuda_sdkdir())
--     if cuda then
--         local linkdirs = cuda["linkdirs"]
--         local bin_dir = target:targetdir()
--         if is_plat("windows") then
--             for i, v in ipairs(linkdirs) do
--                 os.cp(path.join(v, "cudadevrt.lib"), bin_dir)
--             end
--         end
--         -- TODO: linux
--     end
-- end)
target_end()

target("lc-nvrtc")
_config_project({
    project_kind = "binary",
    runtime = "MT"
})
if is_plat("windows") then
    add_syslinks("Ws2_32", "User32")
end
set_basename("luisa_nvrtc")
add_deps("lc-cuda-base")
add_links("nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static")
add_files("cuda_nvrtc_compiler.cpp")
target_end()
