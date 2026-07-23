target("tokenizer")
add_deps("lc-backends-dummy", {inherit = false, links = false})
_config_project({
    project_kind = "binary"
})
add_files('*.cpp')
add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-yyjson")
target_end()
