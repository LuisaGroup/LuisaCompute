target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-runtime", "lc-backend")
add_files("**.cpp")
-- add_includedirs("#")
add_includedirs("$(env CUDA_PATH)/include")
add_linkdirs("$(env CUDA_PATH)/lib/x64")
add_links("cuda")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp("src/backends/cuda/cuda_builtin", path.join(binDir, ".data/"))
end)
if is_plat("windows") then	
	add_defines("NOMINMAX", "UNICODE")
	add_syslinks("Cfgmgr32", "Advapi32", "nvrtc")
end