_config_project({
    project_name = "lc-ast",
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-core", "lc-vstl")
add_files("**.cpp")
add_defines("LC_AST_EXPORT_DLL")
