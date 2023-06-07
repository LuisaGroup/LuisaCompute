target("lc-validation-layer")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl")
add_files("**.cpp")
add_headerfiles("**.h")
target_end()
