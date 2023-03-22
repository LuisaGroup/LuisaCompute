target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
on_load(function(target)
	local cuda_path = os.getenv("CUDA_PATH")
	if cuda_path then
		target:add("includedirs", path.join(cuda_path, "include/"))
		target:add("linkdirs", path.join(cuda_path, "lib/x64/"))
		target:add("links", "cuda", "nvrtc")
	else
		target:set("enabled", false)
	end
end)
add_deps("lc-runtime")
add_files("**.cpp", "../common/default_binary_io.cpp", "../common/string_scratch.cpp")
-- add_includedirs("#")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp("src/backends/cuda/cuda_builtin", path.join(binDir, ".data/"))
end)
if is_plat("windows") then	
	add_defines("NOMINMAX", "UNICODE")
	add_syslinks("Cfgmgr32", "Advapi32")
end