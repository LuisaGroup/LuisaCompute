_config_project({
	project_name = "lc-api",
	project_kind = "shared"
})
add_deps("lc-ir")
add_files("**.cpp")
add_includedirs("../rust")