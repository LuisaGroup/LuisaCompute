target("lc-backend-dx")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-builtin")
add_files("**.cpp", "../common/hlsl/*.cpp")
add_headerfiles("**.h", "../common/default_binary_io.h",
				"../common/hlsl/*.h")
add_includedirs("./")
add_syslinks("D3D12", "dxgi")
if is_plat("windows") then
	add_defines("UNICODE", "_CRT_SECURE_NO_WARNINGS")
end
on_load(function(target)
	local cuda_path = os.getenv("CUDA_PATH")
	if cuda_path then
		target:add("includedirs", path.join(cuda_path, "include/"))
		target:add("linkdirs", path.join(cuda_path, "lib/x64/"))
		target:add("links", "nvrtc", "cudart", "cuda")
		target:add("defines", "LCDX_ENABLE_CUDA")
		if is_plat("windows") then
			target:add("syslinks", "Cfgmgr32", "Advapi32")
		end
	end
end)
set_pcxxheader("pch.h")
add_rules('lc_install_sdk', {libnames = {'dx_sdk'}})
add_packages("zlib", {
    public = false,
    inherit = false
})
target_end()
