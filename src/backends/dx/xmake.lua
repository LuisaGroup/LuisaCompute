target("lc-backend-dx")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_files("**.cpp")
add_headerfiles("**.h", "../common/default_binary_io.h")
add_includedirs("./")
add_syslinks("dxgi")
if is_plat("windows") then
    add_defines("UNICODE", "_CRT_SECURE_NO_WARNINGS")
end
on_load(function(target)
    local dx_sdk = get_config("lc_dx_sdk_dir")
    if dx_sdk then
        target:add("linkdirs", dx_sdk)
        target:add("links", "D3D12")
        target:add("defines", "LUISA_DX_SDK")
    else
        target:add("syslinks", "D3D12")
    end
    if get_config("lc_enable_win_pix") then
        target:add("linkdirs", target:targetdir())
        target:add("links", "WinPixEventRuntime")
        target:add("defines", "LCDX_ENABLE_WINPIX")
    end
    if get_config("lc_enable_dxagsdk") then
        target:add("defines", "LCDX_ENABLE_AGILITY_SDK")
    end
    if get_config("lc_backend_lto") then
        target:set("policy", "build.optimization.lto", true)
        if get_config("lc_toolchain") == "llvm" then
            target:add("ldflags", "-fuse-ld=lld-link")
            target:add("shflags", "-fuse-ld=lld-link")
        end
    end
    if get_config("lc_dx_cuda_interop") then
        import("detect.sdks.find_cuda")
        import("cuda_sdkdir", {
            rootdir = path.translate(path.join(os.scriptdir(), "../cuda"))
        })
        local cuda = find_cuda(cuda_sdkdir())

        local cuda_path = os.getenv("CUDA_PATH")
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
        if not cuda then
            utils.error("cuda not found.")
        else
            local cuda_linkdirs = cuda["linkdirs"]
            set("linkdirs", cuda_linkdirs)
            set("includedirs", cuda["includedirs"])
            if is_plat("linux") and type(cuda_linkdirs) == "table" then
                for _, v in ipairs(cuda_linkdirs) do
                    local stubs_dir = path.join(v, "stubs")
                    if os.exists(stubs_dir) then
                        target:add("linkdirs", stubs_dir, {
                            public = true
                        })
                    end
                end
            end

            target:add("links", "nvrtc_static", "cudart_static", "cuda")
            target:add("defines", "LCDX_ENABLE_CUDA")
            target:add("syslinks", "Cfgmgr32", "Advapi32")
        end
    end
end)
set_pcxxheader("lc_dx_pch.h")
add_rules('lc_install_sdk', {
    libnames = {'dx_sdk'}
})
target_end()
