_config_project({
	project_name = "lc-backend-dx",
	project_kind = "shared",
	batch_size = 8,
	is_backend = true
})
if is_mode("debug") then
	add_defines("SHADER_COMPILER_TEST")
end
add_deps("lc-runtime", "lc-vstl")
add_files("Api/**.cpp", "DXRuntime/**.cpp", "Resource/**.cpp", "Shader/**.cpp", "HLSL/**.cpp")
add_includedirs("./")
add_syslinks("D3D12", "dxgi")
after_build(function(target)
	local binDir = target:targetdir() .. '/'
	os.cp("src/backends/dx/dx_builtin", binDir .. ".data/")
	os.cp("src/backends/dx/dx_support/*.dll", binDir)
end)