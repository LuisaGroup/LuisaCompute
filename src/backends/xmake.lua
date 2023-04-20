if LCDxBackend or LCVkBackend then
	target("lc-copy-dxc")
	set_kind("phony")
	after_build(function(target)
		if is_plat("windows") then
			local bin_dir = target:targetdir()
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
target("lc-backends-dummy")
set_kind("phony")
add_deps("lc-validation-layer", {inherit = false})
if LCDxBackend then
	add_deps("lc-backend-dx", {inherit = false})
end
if LCMetalBackend then
	add_deps("lc-backend-metal", {inherit = false})
end
if LCVkBackend then
	add_deps("lc-backend-vk", {inherit = false})
end
if LCVkBackend or LCDxBackend then
	add_deps("lc-hlsl-builtin", {inherit = false})
end
if LCCpuBackend then
	add_deps("lc-backend-cpu", {inherit = false})
end
target_end()