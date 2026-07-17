target("lc-backend-dx")
set_basename("luisa-backend-dx")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
add_files("**.cpp")
add_headerfiles("**.h")
add_includedirs("./")
add_syslinks("dxgi")
if is_plat("windows") then
    add_defines("NOMINMAX", "UNICODE", "_CRT_SECURE_NO_WARNINGS")
end
	on_load(function(target)
	    target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
	    target:add("syslinks", "D3D12", "dcomp")
	    target:add("defines", "LUISA_DX_SDK")
	    if has_config("lc_enable_win_pix") then
	        target:add("linkdirs", target:targetdir())
	        target:add("links", "WinPixEventRuntime")
	        target:add("defines", "LCDX_ENABLE_WINPIX")
	    end
	    if has_config("lc_dx_cuda_interop") then
	        import("detect.sdks.find_cuda")
	        target:add("links", "nvrtc_static", "cudart_static", "cuda")
	        target:add("defines", "LCDX_ENABLE_CUDA")
	        target:add("syslinks", "Cfgmgr32", "Advapi32")
	        target:add('deps', '_lc_cuda_base')
	    end
	end)
lc_set_pcxxheader("lc_dx_pch.h")
target_end()
