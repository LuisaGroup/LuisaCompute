if LCDxBackend or LCVkBackend then
	target("lc-copy-dxc")
	set_kind("phony")
	after_build(function(target)
		local bin_dir = target:targetdir()
		os.cp(path.join(os.scriptdir(), "dx/dx_builtin"), path.join(bin_dir, ".data"))
		os.cp(path.join(os.scriptdir(), "dx/dx_support/*.dll"), bin_dir)
	end)
	target_end()
end

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
