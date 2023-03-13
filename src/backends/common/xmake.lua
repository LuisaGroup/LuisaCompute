target("lc-backend")
_config_project({
	project_kind = "static"
})
add_files("**.cpp")
add_deps("lc-runtime")