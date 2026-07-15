set_xmakever("3.0.6")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)
-- pre-defined options
-- enable mimalloc as default allocator: https://github.com/LuisaGroup/mimalloc
option("lc_enable_mimalloc", {default = true})
-- enable custom memory allocator instead of default system allocator
option("lc_enable_custom_malloc", {default = false})
-- enable unity(jumbo) build, enable this option will optimize compile speed
option("lc_enable_unity_build", {default = true})
-- enable Pre-Compile header, enable this option will optimize compile speed
option("lc_enable_pch", {default = true})
-- enable sse and sse2 SIMD
option("lc_enable_simd", {default = true})
-- enable DirectX-12 backend
option("lc_dx_backend", {default = true})
-- enable fallback CPU backend for platforms without GPU support
option("lc_fallback_backend", {default = false})
-- enable XIR (Intermediate Representation) support
option("lc_enable_xir", {default = false})
-- use external Marl library instead of bundled version
option("lc_external_marl", {default = false})
-- enable DirectX-CUDA interoperability
option("lc_dx_cuda_interop", {default = false})
-- enable Vulkan-CUDA interoperability
option("lc_vk_cuda_interop", {default = false})
-- enable Link Time Optimization (LTO) for smaller binary size
option("lc_use_lto", {default=false})
-- enable Vulkan backend
option("lc_vk_backend", {default = true})

option("lc_vk_backend_use_xir_spirv", {default = false})
-- enable toy C backend for testing and debugging
option("lc_toy_c_backend", {default = false})
-- enable NVIDIA-CUDA backend
option("lc_cuda_backend", {default = true})
-- enable NVIDIA-CUDA Extension CUB, default false, because of long compile time
option("lc_cuda_ext_lcub", {default = false})
-- enable Metal backend
option("lc_metal_backend", {default = true})
-- enable tests module
option("lc_enable_tests", {default = true})
-- C++ standard version (e.g., cxx17, cxx20, cxx23)
option("lc_cxx_standard", {default = 'cxx20'})
-- C standard version (e.g., c11, clatest)
option("lc_c_standard", {default = 'clatest'})
-- enable C++ Run-Time Type Information (RTTI)
option("lc_rtti", {default = false})
-- enable safe mode for runtime
option("lc_safe_mode", {default = false})

-- Python include directory path
option("lc_py_include", {default = false})
-- Python library directory path
option("lc_py_linkdir", {default = false})
-- Python libraries to link
option("lc_py_libs", {default = false})
-- enable OpenShadingLanguage (OSL) support
option("lc_enable_osl", {default = true})
-- enable C++ DSL module
option("lc_enable_dsl", {default = true})
-- enable clang C++ module
option("lc_enable_clangcxx", {default = false})
-- enable GUI module
option("lc_enable_gui", {default = true})
-- enable ImGui integration for GUI
option("lc_enable_imgui", {default = true})
-- custom binary output directory
option("lc_bin_dir", {default = "bin"})
-- enable Python bindings
option("lc_enable_py", {default = true})
-- download and extract package dir, set to empty string to disable installation
option("lc_sdk_dir", {default = false})
-- custom toolchain path or name
option("lc_toolchain", {default = false})
-- Windows runtime library (MT/MD/MTd/MDd)
option("lc_win_runtime", {default = false})
-- additional optimization flags
option("lc_optimize", {default = false})
-- custom LLVM installation path
option("lc_llvm_path", {default = false})
-- custom Embree installation path
option("lc_embree_path", {default = false})
-- Use AST→LLVM→SPIR-V codegen for Vulkan backend (mutually exclusive with lc_vk_backend_use_xir_spirv)
option("lc_vk_backend_use_ast_llvm_spirv", {default = false,
    description = "Use AST LLVM codegen for Vulkan backend SPIR-V generation.",
    showmenu = true})
-- use system STL instead of bundled or custom one
option("lc_use_system_stl", {default = false})
-- third-party: use xmake-repo packages instead of bundled sources
-- use xmake-repo spdlog package instead of bundled
option("lc_spdlog_use_xrepo", {default = false})
-- use xmake-repo reproc package instead of bundled
option("lc_reproc_use_xrepo", {default = false})
-- use xmake-repo lmdb package instead of bundled
option("lc_lmdb_use_xrepo", {default = false})
-- use xmake-repo imgui package instead of bundled
option("lc_imgui_use_xrepo", {default = false})
-- use xmake-repo glfw package instead of bundled
option("lc_glfw_use_xrepo", {default = false})
-- use xmake-repo yyjson package instead of bundled
option("lc_disable_win_message_box", {default = true})

option("lc_yyjson_use_xrepo", {default = false})
-- internal: xmake scripts directory path
option("lc_scripts_path")

function lc_set_pcxxheader(...)
    if get_config('lc_enable_pch') then
        set_pcxxheader(...)
    end
end

set_showmenu(false)
set_default(false)
after_check(function(option)
    option:set_value(path.join(os.scriptdir(), 'scripts'))
end)
option_end()
-- internal: external dependencies directory path
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
if has_config("lc_vk_backend_use_xir_spirv") and has_config("lc_vk_backend_use_ast_llvm_spirv") then
    raise("Vulkan compute shader codegen options are mutually exclusive. " ..
          "Use lc_vk_backend_use_xir_spirv for the default native SPIR-V path, " ..
          "or lc_vk_backend_use_ast_llvm_spirv for the experimental LLVM SPIR-V path.")
end
includes("scripts/xmake_func.lua")

if has_config('_lc_check_env') then
    local lc_bin_dir = get_config("_lc_bin_dir")
    if lc_bin_dir then
        set_targetdir(lc_bin_dir)
    end
    includes("src")
    if has_config("lc_enable_tests") then
        includes("examples")
        includes("tutorials")
    end
end
