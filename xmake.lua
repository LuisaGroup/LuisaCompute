set_xmakever("3.0.2")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)
-- pre-defined options
-- enable mimalloc as default allocator: https://github.com/LuisaGroup/mimalloc
option("lc_enable_mimalloc")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("lc_enable_custom_malloc")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

-- enable unity(jumbo) build, enable this option will optimize compile speed
option("lc_enable_unity_build")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable sse and sse2 SIMD
option("lc_enable_simd")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable DirectX-12 backend
option("lc_dx_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("lc_dx_sdk_dir")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("lc_external_marl")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("lc_dx_cuda_interop")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("lc_vk_cuda_interop")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

-- enable Vulkan backend
option("lc_vk_support")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("lc_vk_backend")
set_values(true, false)
-- TODO: vulkan backend not ready
set_default(false)
set_showmenu(true)
option_end()

option("lc_toy_c_backend")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable NVIDIA-CUDA backend
option("lc_cuda_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable NVIDIA-CUDA Extension CUB
option("lc_cuda_ext_lcub")
set_values(true, false)
set_default(false) -- default false, because of long compile time
set_showmenu(true)
option_end()
-- enable CPU backend
option("lc_cpu_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable Metal backend
option("lc_metal_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable tests module
option("lc_enable_tests")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- python include path
option("lc_py_include")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("lc_py_linkdir")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("lc_py_libs")
set_default(false)
set_showmenu(true)
option_end()
-- enable intermediate representation module (rust required)
option("lc_enable_ir")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable osl
option("lc_enable_osl")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable c-language api module for cross-language bindings module
option("lc_enable_api")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable C++ DSL module
option("lc_enable_dsl")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable clang C++ module
option("lc_enable_clangcxx")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable GUI module
option("lc_enable_gui")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- custom bin dir
option("lc_bin_dir")
set_default("bin")
set_showmenu(true)
option_end()
-- custom sdk dir
option("lc_sdk_dir")
set_default(false)
set_showmenu(true)
option_end()

option("lc_toolchain")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("lc_win_runtime")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("lc_llvm_path")
set_default(false)
set_showmenu(true)
option_end()

option("lc_use_system_stl")
set_default(false)
set_showmenu(true)
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

if get_config('_lc_check_env') then
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