-- lc_compile_builtin: AOT-compile a raw HLSL builtin source into the bytecode
-- containers read by the DX/VK builtin-kernel loaders, and into the embedded
-- Vulkan device-library arrays the backend build generates.
--
-- Usage:
-- xmake build lc_compile_builtin
-- xmake run lc_compile_builtin <dx|vk|spv> <input-shader> <output> [options]
-- xmake run lc_compile_builtin <dx|vk> inspect <artifact>
-- xmake run lc_compile_builtin spv embed <module.spv>... -o <embedded.cpp> -h <embedded.h>
-- e.g.:
-- xmake run lc_compile_builtin dx src/backends/common/hlsl/builtin/bindless_upload.bytes src/backends/common/hlsl/builtin/load_bdls.dxil
-- xmake run lc_compile_builtin vk src/backends/common/hlsl/builtin/bindless_upload.bytes src/backends/common/hlsl/builtin/load_bdls_vk.dxil
-- xmake run lc_compile_builtin spv src/backends/vk/builtin/accel_process.comp.hlsl accel_process.spv
--
-- The `dx`/`vk` routes drive dxc (dxcompiler.dll/dxil.dll) from the build output
-- directory at run time; the `spv` route drives the same host tools the Vulkan
-- backend build uses (luisa-glslang, luisa-validate-spirv, luisa-embed-device-lib),
-- which are resolved from that same runtime directory.
target("lc_compile_builtin")
    set_basename("lc-compile-builtin")
    set_kind("binary")
    set_default(false)
    _config_project({
        project_kind = "binary"
    })
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen")
    -- The glslang/SPIRV-Tools/embedder host tools are spawned as subprocesses.
    if has_config("lc_reproc_use_xrepo") then
        add_packages("reproc")
    else
        add_deps("reproc")
    end
    if is_plat("windows") then
        add_syslinks("d3d12")
    end
    add_files("main.cpp")
    add_includedirs("./")
target_end()
