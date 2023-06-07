target("lc-api")
_config_project({
	project_kind = "shared"
})
add_deps("lc-ir")
add_headerfiles("**.h")
add_files("**.cpp")
add_includedirs("../rust")
target_end()
