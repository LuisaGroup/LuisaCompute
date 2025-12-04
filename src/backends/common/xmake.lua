if (has_config("lc_vk_backend") or has_config("lc_dx_backend")) then
    includes("hlsl")
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

if has_config("lc_toy_c_backend") then
    target("lc-clanguage-codegen")
    set_basename("luisa-clanguage-codegen")
    _config_project({
        project_kind = "static"
    })
    add_deps("lc-core", "lc-runtime", "lc-vstl")
    add_files("c_codegen/*.cpp", "hlsl/string_builder.cpp")
    set_pcxxheader("c_codegen/lc_ccodegen_pch.h")
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
