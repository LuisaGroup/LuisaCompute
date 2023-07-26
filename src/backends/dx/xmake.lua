target("lc-backend-dx")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-runtime", "lc-vstl")
add_files("**.cpp")
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
after_build(function(target)
	if is_plat("windows") then
		local bin_dir = target:targetdir()
		os.cp(path.join(os.scriptdir(), "dx_support/*.dll"), bin_dir)
	end
end)
set_pcxxheader("pch.h")
target_end()
