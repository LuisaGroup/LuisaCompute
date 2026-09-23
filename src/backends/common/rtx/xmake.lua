-- luisa-fallback-rtx: the software (hand-built two-level LBVH) ray-tracing
-- fallback shared by the backends that have no hardware ray tracing, or that the
-- user asked to run in fallback mode.  It is backend-independent: it only needs
-- the runtime and the DSL, and it talks to the backend through the
-- `DeviceInterface` it is handed (fallback_rtx.h).
--
-- This file is included by src/backends/common/xmake.lua when at least one of
-- the cuda/dx/vk backends is enabled.

target("luisa-fallback-rtx")
set_basename("luisa-fallback-rtx")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-runtime", "lc-dsl", "lc-vstl")
add_files("fallback_rtx_sort.cpp", "fallback_rtx_storage.cpp", "fallback_rtx_blas.cpp",
          "fallback_rtx_tlas.cpp", "fallback_rtx_device.cpp",
          "fallback_rtx_layout_contract.cpp")
add_headerfiles("*.h")
-- The public surface is `fallback_rtx.h`, so the directory itself is an include
-- directory of every backend that depends on this target.
add_includedirs(".", {
    public = true
})
target_end()

-- The self-check: creates a real device, builds a two-level LBVH through the
-- library's public API only, and validates it.  It is a plain binary (not part
-- of the backend libraries) so `xmake run fallback_rtx_selfcheck` can always be
-- used to answer "is the fallback build right on this machine?".
target("fallback_rtx_selfcheck")
-- Built after every backend, but linked against none of them: the device is
-- created at runtime through `Context::create_device`.
add_deps("lc-backends-dummy", {
    inherit = false,
    links = false
})
_config_project({
    project_kind = "binary"
})
add_files("fallback_rtx_selfcheck.cpp")
add_deps("luisa-fallback-rtx")
target_end()
