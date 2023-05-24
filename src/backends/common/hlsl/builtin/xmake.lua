target("lc-hlsl-builtin")
_config_project({
    project_kind = "shared",
    batch_size = 64
})
add_files("*.c")
add_defines("LC_HLSL_DLL")
target_end()