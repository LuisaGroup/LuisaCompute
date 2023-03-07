target("lc-backend-dx")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
if is_mode("debug") then
	add_defines("SHADER_COMPILER_TEST")
end
add_deps("lc-runtime", "lc-vstl")
add_files("DXApi/**.cpp", "DXRuntime/**.cpp", "Resource/**.cpp", "Shader/**.cpp", "HLSL/**.cpp")
add_includedirs("./")
add_syslinks("D3D12", "dxgi")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp("src/backends/dx/dx_builtin", path.join(binDir, ".data/"))
	os.cp("src/backends/dx/dx_support/*.dll", binDir)
end)
