target("lc-dsl")
_config_project({
	project_kind = "static",
	batch_size = 16
})
set_pcxxheader("pch.h")
add_deps("lc-ast", "lc-runtime")
add_headerfiles("../../include/luisa/dsl/**.h")
add_files("**.cpp")
add_defines("LUISA_DSL_STATIC_LIB", {
    public = true
})