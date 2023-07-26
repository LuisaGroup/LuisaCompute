target("lc-validation-layer")
_config_project({
	project_kind = "shared"
})
set_pcxxheader("pch.h")
add_deps("lc-runtime", "lc-vstl")
add_files("**.cpp")
add_headerfiles("**.h")
target_end()
