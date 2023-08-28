target("lc-ast")
_config_project({
	project_kind = "shared"
})
add_deps("lc-core", "lc-vstl")
add_headerfiles("../../include/luisa/ast/**.h")
set_pcxxheader("pch.h")
add_files("**.cpp")
add_cxflags("/bigobj", {
	tools = "cl"
})
add_defines("LC_AST_EXPORT_DLL")
target_end()
