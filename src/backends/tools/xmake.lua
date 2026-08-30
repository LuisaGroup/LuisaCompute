-- lc_compile_builtin: AOT-compile builtin kernels to .dxil (DX) or .spv (VK).
--
-- Usage:
--   xmake build lc_compile_builtin
--   xmake run lc_compile_builtin <backend> <destination_dir>
--   e.g.: xmake run lc_compile_builtin dx ./builtin_out
--         xmake run lc_compile_builtin vk ./builtin_out
--
-- Creates a headless device and uses compile_only to save shader bytecode.

target("lc_compile_builtin")
    set_basename("lc-compile-builtin")
    set_kind("binary")
    set_default(false)
    _config_project({
        project_kind = "binary"
    })
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    add_deps("lc-runtime", "lc-dsl", "lc-vstl")
    add_files("main.cpp")
    add_includedirs("./")
target_end()
