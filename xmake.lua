set_xmakever("2.8.7")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)
-- pre-defined options
-- enable mimalloc as default allocator: https://github.com/LuisaGroup/mimalloc
option("enable_custom_malloc")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
option("enable_mimalloc")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable unity(jumbo) build, enable this option will optimize compile speed
option("enable_unity_build")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable sse and sse2 SIMD
option("enable_simd")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable DirectX-12 backend
option("dx_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable Vulkan backend
option("vk_backend")
set_values(true, false)
-- TODO: vulkan backend not ready
set_default(false)
set_showmenu(true)
option_end()
-- enable NVIDIA-CUDA backend
option("cuda_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable NVIDIA-CUDA Extension CUB
option("cuda_ext_lcub")
set_values(true, false)
set_default(false) -- default false, because of long compile time
set_showmenu(true)
option_end()
-- enable CPU backend
option("cpu_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable Metal backend
option("metal_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable tests module
option("enable_tests")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- python include path
option("py_include")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("py_linkdir")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("py_libs")
set_default(false)
set_showmenu(true)
option_end()
-- enable intermediate representation module (rust required)
option("enable_ir")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable osl
option("enable_osl")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable c-language api module for cross-language bindings module
option("enable_api")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable C++ DSL module
option("enable_dsl")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable clang C++ module
option("enable_clangcxx")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable GUI module
option("enable_gui")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- custom bin dir
option("bin_dir")
set_default("bin")
set_showmenu(true)
option_end()
-- custom sdk dir
option("sdk_dir")
set_default(false)
set_showmenu(true)
option_end()
-- external_marl
option("external_marl")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
option("lc_toolchain")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- pre-defined options end

-- use xrepo from skr
add_repositories("skr-xrepo xrepo", {
    rootdir = os.projectdir()
})
-- try options.lua
if path.absolute(os.projectdir()) == path.absolute(os.scriptdir()) and os.exists("scripts/options.lua") then
    includes("scripts/options.lua")
end
if lc_toolchain then
    for k, v in pairs(lc_toolchain) do
        set_config(k, v)
    end
end
includes("scripts/xmake_func.lua")

if get_config('_lc_check_env') then
    local bin_dir = get_config("_lc_bin_dir")
    if bin_dir then
        set_targetdir(bin_dir)
    end
    includes("src")
end
