target("lc-osl")
set_basename("luisa-osl")
_config_project({
	project_kind = "shared",
	batch_size = 16
})
add_defines("LUISA_OSL_EXPORT_DLL")
add_deps("lc-runtime")
add_headerfiles("../../include/luisa/osl/**.h")
add_files(
    "hint.cpp",
    "instruction.cpp",
    "literal.cpp",
    "oso_parser.cpp",
    "shader.cpp",
    "symbol.cpp",
    "type.cpp"
)
if has_config("lc_enable_xir") then
    add_defines("LUISA_ENABLE_XIR")
    add_files(
        "string_table.cpp",
        "shader_group.cpp",
        "string_analysis.cpp",
        "deriv_analysis.cpp",
        "translator.cpp",
        "translator_ops.cpp",
        "translator_color.cpp",
        "translator_noise.cpp"
    )
end
