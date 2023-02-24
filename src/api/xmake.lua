_config_project({
	project_name = "lc-api",
	project_kind = "shared"
})
local add_includedirs = _get_add_includedirs()
local add_defines = _get_add_defines()
add_deps("lc-ir")
add_files("**.cpp")
add_includedirs("../rust")