target("lc-backend-dx")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-runtime", "lc-vstl")
add_files("DXApi/**.cpp", "DXRuntime/**.cpp", "Resource/**.cpp", "Shader/**.cpp", "HLSL/**.cpp",
				"../common/default_binary_io.cpp")
add_includedirs("./")
add_syslinks("D3D12", "dxgi")
add_deps("lc-copy-dxc")
if is_plat("windows") then
	add_defines("NOMINMAX", "UNICODE")
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
target_end()
