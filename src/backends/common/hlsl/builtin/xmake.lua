target("lc-hlsl-builtin")
_config_project({
    project_kind = "static",
    batch_size = 64
})
add_files("*.cpp")
add_defines("LC_HLSL_DLL")
target_end()