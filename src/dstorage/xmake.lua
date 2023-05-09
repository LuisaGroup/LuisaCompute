-- direct storage currently only for Windows OS
target("lc-dstorage")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime")
add_files("*.cpp")
after_build(function(target)
	local bin_dir = target:targetdir()
	os.cp(path.join(os.scriptdir(), "bin/*.dll"), bin_dir)
end)
add_syslinks("D3D12", "OleAut32", "Ole32")
target_end()
