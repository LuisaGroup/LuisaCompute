-- lc_compile_builtin: AOT-compile a raw HLSL builtin source into the bytecode
-- containers read by the DX/VK builtin-kernel loaders.
--
-- Usage:
--   xmake build lc_compile_builtin
--   xmake run lc_compile_builtin <dx|vk> <input-hlsl> <output> [options]
-- e.g.:
--   xmake run lc_compile_builtin dx src/backends/common/hlsl/builtin/bindless_upload.bytes src/backends/common/hlsl/builtin/load_bdls.dxil
--   xmake run lc_compile_builtin vk src/backends/common/hlsl/builtin/bindless_upload.bytes src/backends/common/hlsl/builtin/load_bdls_vk.dxil
--
-- The dxc compiler and its runtime libraries (dxcompiler.dll/dxil.dll) are
-- loaded from the build output directory at run time.
target("lc_compile_builtin")
    set_basename("lc-compile-builtin")
    set_kind("binary")
    set_default(false)
    _config_project({
        project_kind = "binary"
    })
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
    if is_plat("windows") then
        add_syslinks("d3d12")
    end
    add_files("main.cpp")
    add_includedirs("./")
target_end()
