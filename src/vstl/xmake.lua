target("lc-vstl")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
add_deps("lc-core")
add_files("**.cpp")
add_defines("LC_VSTL_EXPORT_DLL")
if is_plat("windows") then
	add_syslinks("Ole32")
elseif is_plat("macosx") then
	add_frameworks("CoreFoundation")
end
