target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-runtime", "lc-vstl")
add_files("**.cpp")
add_defines("NOMINMAX", "UNICODE")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp("src/backends/cuda/cuda_builtin", path.join(binDir, ".data/"))
end)
