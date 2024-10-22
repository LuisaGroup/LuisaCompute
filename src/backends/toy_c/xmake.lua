target("lc-backend-toy-c")
_config_project({
	project_kind = "shared"
})
add_deps("lc-runtime", "lc-vstl", "lc-clanguage-codegen")
add_files("*.cpp")
add_headerfiles("**.h")
set_pcxxheader("pch.h")
target_end()
