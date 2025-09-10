target("lc-backend-dx")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_files("**.cpp")
add_headerfiles("**.h", "../common/default_binary_io.h")
add_includedirs("./")
add_syslinks("D3D12", "dxgi")
if is_plat("windows") then
    add_defines("UNICODE", "_CRT_SECURE_NO_WARNINGS")
end
on_load(function(target)
    if get_config("enable_win_pix") then
        target:add("linkdirs", target:targetdir())
        target:add("links", "WinPixEventRuntime")
        target:add("defines", "LCDX_ENABLE_WINPIX")
    end
    if get_config("enable_dxagsdk") then
        target:add("defines", "LCDX_ENABLE_AGILITY_SDK")
    end
    if get_config("lc_backend_lto") then
        target:set("policy", "build.optimization.lto", true)
        if get_config("lc_toolchain") == "llvm" then
            target:add("ldflags", "-fuse-ld=lld-link")
            target:add("shflags", "-fuse-ld=lld-link")
        end
    end
    if get_config("dx_cuda_interop") then
        local cuda_path = os.getenv("CUDA_PATH")
        if not cuda_path then
            utils.error("CUDA_PATH not found.")
        else
            target:add("includedirs", path.join(cuda_path, "include/"))
            target:add("linkdirs", path.join(cuda_path, "lib/x64/"))
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
