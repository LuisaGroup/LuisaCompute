target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
add_rules("lc_vulkan")
add_deps("lc-runtime")
add_files("**.cpp", "../common/default_binary_io.cpp", "../common/string_scratch.cpp")
on_load(function(target)
	local cuda_path = os.getenv("CUDA_PATH")
	if cuda_path then
		target:add("includedirs", path.join(cuda_path, "include/"))
		target:add("linkdirs", path.join(cuda_path, "lib/x64/"))
		target:add("links", "nvrtc", "cudart", "cuda")
	else
		target:set("enabled", false)
		return
	end
	if is_plat("windows") then
		target:add("defines", "NOMINMAX", "UNICODE")
		target:add("syslinks", "Cfgmgr32", "Advapi32")
	end
end)
-- add_includedirs("#")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp("src/backends/cuda/cuda_builtin", path.join(binDir, ".data/"))
end)