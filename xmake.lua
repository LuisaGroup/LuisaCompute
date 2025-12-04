set_xmakever("3.0.5")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)

-- pre-defined options
-- enable mimalloc as default allocator: https://github.com/LuisaGroup/mimalloc
option("lc_enable_mimalloc", {default = true})
option("lc_enable_custom_malloc", {default = false})
-- enable unity(jumbo) build, enable this option will optimize compile speed
option("lc_enable_unity_build", {default = true})
-- enable sse and sse2 SIMD
option("lc_enable_simd", {default = true})
-- enable DirectX-12 backend
option("lc_dx_backend", {default = true})
option("lc_fallback_backend", {default = false})
option("lc_enable_xir", {default = false})
option("lc_external_marl", {default = false})
option("lc_dx_cuda_interop", {default = false})
option("lc_vk_cuda_interop", {default = false})
-- enable Vulkan backend
option("lc_vk_backend", {default = true})
option("lc_toy_c_backend", {default = false})
-- enable NVIDIA-CUDA backend
option("lc_cuda_backend", {default = true})
-- enable NVIDIA-CUDA Extension CUB, default false, because of long compile time
option("lc_cuda_ext_lcub", {default = false})
-- enable Metal backend
option("lc_metal_backend", {default = true})
-- enable tests module
option("lc_enable_tests", {default = true})
-- python include path
option("lc_py_include", {default = false})
-- python include path
option("lc_py_linkdir", {default = false})
-- python include path
option("lc_py_libs", {default = false})
-- enable osl
option("lc_enable_osl", {default = true})
-- enable C++ DSL module
option("lc_enable_dsl", {default = true})
-- enable clang C++ module
option("lc_enable_clangcxx", {default = false})
-- enable GUI module
option("lc_enable_gui", {default = true})
-- custom bin dir
option("lc_bin_dir", {default = "bin"})
-- custom sdk dir
option("lc_enable_py", {default = true})
option("lc_sdk_dir", {default = false})
option("lc_toolchain", {default = false})
option("lc_win_runtime", {default = false})
option("lc_llvm_path", {default = false})
option("lc_embree_path", {default = false})
option("lc_use_system_stl", {default = false})
-- third-party
option("lc_spdlog_use_xrepo", {default = false})
option("lc_reproc_use_xrepo", {default = false})
option("lc_lmdb_use_xrepo", {default = false})
option("lc_imgui_use_xrepo", {default = false})
option("lc_glfw_use_xrepo", {default = false})
option("lc_yyjson_use_xrepo", {default = false})
-- xmake file paths
option("lc_scripts_path")
set_showmenu(false)
set_default(false)
after_check(function(option)
    option:set_value(path.join(os.scriptdir(), 'scripts'))
end)
option_end()
option("lc_ext_path")
set_showmenu(false)
set_default(false)
after_check(function(option)
    option:set_value(path.join(os.scriptdir(), 'src/ext'))
end)
option_end()

-- pre-defined options end
-- try options.lua
if path.absolute(os.projectdir()) == path.absolute(os.scriptdir()) and os.exists("scripts/options.lua") then
    includes("scripts/options.lua")
end
if lc_options then
    for k, v in pairs(lc_options) do
        set_config(k, v)
    end
end
includes("scripts/xmake_func.lua")

if has_config('_lc_check_env') then
    local lc_bin_dir = get_config("_lc_bin_dir")
    if lc_bin_dir then
        set_targetdir(lc_bin_dir)
    end
    includes("src")
end

target("lc_embed_codegen")
    add_rules("lc_basic_settings", {
        project_kind = "binary"
    })
    add_files("utils/embed_codegen.cpp")
    set_policy("build.fence", true)
target_end()
