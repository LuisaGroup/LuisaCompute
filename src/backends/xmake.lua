if LCDxBackend or LCVkBackend then
	target("lc-copy-dxc")
	set_kind("phony")
	after_build(function(target)
		if is_plat("windows") then
			local bin_dir = target:targetdir()
			local data_dir = path.join(bin_dir, ".data/dx_builtin")
			os.mkdir(data_dir)
			os.cp(path.join(os.scriptdir(), "dx/dx_builtin/*"), data_dir)
			os.cp(path.join(os.scriptdir(), "dx/dx_support/*.dll"), bin_dir)
		end
	end)
	target_end()
end
includes("common")
if LCDxBackend then
	includes("dx")
end
if LCCudaBackend then
	includes("cuda")
end
if LCMetalBackend then
	includes("metal")
end
if LCCpuBackend then
	includes("cpu")
end
if LCRemoteBackend then
	includes("remote")
end
if LCVkBackend then
	includes("vk")
end
includes("validation")
