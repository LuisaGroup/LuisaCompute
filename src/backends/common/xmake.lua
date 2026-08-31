if has_config("lc_vk_backend") or has_config("lc_dx_backend") then
    includes("hlsl")
end

-- Both SPIR-V code generators are Vulkan-only. Key their targets on the
-- backend as well as the codegen option so a DX-only/test configuration cannot
-- create dangling dependencies on lc-spirv or lc-spirv-llvm.
if has_config("lc_vk_backend") then
    local use_ast_llvm_spirv =
        has_config("lc_vk_backend_use_ast_llvm_spirv")
    if has_config("lc_vk_backend_use_xir_spirv") or
       use_ast_llvm_spirv then
        includes("spirv")
    end
    if use_ast_llvm_spirv then
        includes("spirv_llvm")
    end
end
if has_config("lc_cuda_backend") then
    target("lc-vulkan-swapchain")
    _config_project({
        project_kind = "object"
    })
    set_values("vk_public", true)
    add_headerfiles("vulkan_instance.h")
    add_files("vulkan_swapchain.cpp", "vulkan_instance.cpp")
    add_deps("lc-core", "lc-volk")
    if is_plat("linux") then
        add_syslinks("xcb", "X11", {
            public = true
        })
    end
    on_load(function(target)
        if target:is_plat("macosx") then
            target:add("files", path.join(os.scriptdir(), "moltenvk_surface.mm"))
        end
    end)
    target_end()
end

if has_config("lc_cuda_backend") or has_config('lc_dx_cuda_interop') or has_config('lc_vk_cuda_interop') then
    target("_lc_cuda_base")
    set_kind('phony')
    on_load(function(target)
        import("cuda_sdkdir", {
            rootdir = get_config('lc_scripts_path')
        })
        import("detect.sdks.find_cuda")
        local cuda = find_cuda(cuda_sdkdir())
        if cuda then
            local cuda_linkdirs = cuda["linkdirs"]
            target:add("linkdirs", cuda_linkdirs, {
                public = true
            })
            if target:is_plat("linux") and type(cuda_linkdirs) == "table" then
                for _, v in ipairs(cuda_linkdirs) do
                    local stubs_dir = path.join(v, "stubs")
                    if os.exists(stubs_dir) then
                        target:add("linkdirs", stubs_dir, {
                            public = true
                        })
                    end
                end
            end
            target:add("includedirs", cuda["includedirs"], {
                public = true
            })
        else
            utils.error('cuda not found.')
            return
        end
    end)
    target_end()
end
