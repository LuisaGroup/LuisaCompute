target("lc-dsl")
_config_project({
	project_kind = "shared",
	batch_size = 16
})
set_pcxxheader("pch.h")
add_defines("LC_DSL_EXPORT_DLL")
add_deps("lc-ast", "lc-runtime")
add_headerfiles("../../include/luisa/dsl/**.h")
add_files("**.cpp")
