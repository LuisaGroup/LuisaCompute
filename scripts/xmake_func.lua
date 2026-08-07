--[[
    xmake_func.lua - Build configuration functions for LuisaCompute
    
    This file defines options, rules, targets and helper functions used across
    the LuisaCompute project build system.
--]] -- ============================================================================
-- SECTION 1: Internal Options
-- ============================================================================
-- Internal option to track Python binding enablement
option("_lc_enable_py", {
    default = false,
    showmenu = false
})

-- Environment validation option
option("_lc_check_env")
set_showmenu(false)
set_default(false)
after_check(function(option)
    -- Validate architecture (only x64 and arm64 are supported)
    if not is_arch("x64", "x86_64", "arm64") then
        option:set_value(false)
        utils.error("Illegal environment. Please check your compiler, architecture or platform.")
        return nil
    end
    -- Validate build mode
    if not (is_mode("debug") or is_mode("release") or is_mode("releasedbg")) then
        option:set_value(false)
        utils.error("Illegal mode. set mode to 'release', 'debug' or 'releasedbg'.")
        return nil
    end
    option:set_value(true)
end)
option_end()

-- Binary output directory configuration option
option("_lc_bin_dir")
set_default(false)
set_showmenu(false)
-- Declare dependencies on all backend and feature options
add_deps("lc_dx_backend", "lc_vk_backend", "lc_cuda_backend", "lc_metal_backend", "lc_enable_tests", "lc_py_include",
    "lc_cuda_ext_lcub", "lc_enable_dsl", "lc_enable_gui", "lc_bin_dir", "lc_dx_cuda_interop", "lc_vk_cuda_interop",
    "_lc_enable_py", "lc_enable_py", "lc_enable_xir", 'lc_vk_backend_use_xir_spirv', 'lc_vk_backend_use_ast_llvm_spirv',
    "lc_fallback_backend", "lc_llvm_path", "lc_embree_path")

before_check(function(option)
    -- Load custom options from options.lua if in project root
    if path.absolute(path.join(os.projectdir(), "scripts")) == path.absolute(os.scriptdir()) then
        local v = import("options", {
            try = true,
            anonymous = true
        })
        if v then
            local opt = v.get_options
            if type(opt) == "function" then
                local map = opt()
                for k, v in pairs(map) do
                    if v ~= nil then
                        option:dep(k):enable(v)
                    end
                end
            end
        end

        -- Set binary directory based on build mode
        local bin_dir = option:dep("lc_bin_dir"):enabled()
        if is_mode("debug") then
            bin_dir = path.join(bin_dir, "debug")
        elseif is_mode("releasedbg") then
            bin_dir = path.join(bin_dir, "releasedbg")
        else
            bin_dir = path.join(bin_dir, "release")
        end
        option:set_value(bin_dir)
    end

    -- Auto-enable DSL when tests are enabled
    local lc_enable_tests = option:dep("lc_enable_tests")
    if lc_enable_tests:enabled() then
        option:dep("lc_enable_dsl"):enable(true, {
            force = true
        })
    end

    -- Check Python binding configuration
    local enable_py = option:dep("_lc_enable_py")
    local function non_empty_str(s)
        return type(s) == "string" and s:len() > 0
    end
    if not option:dep("lc_enable_py"):enabled() then
        enable_py:enable(false)
    elseif non_empty_str(option:dep("lc_py_include"):enabled()) then
        enable_py:enable(true)
    end

    local is_win = is_plat("windows")

    -- Check fallback backend dependencies (requires LLVM and Embree)
    local lc_fallback_backend = option:dep("lc_fallback_backend")
    local lc_llvm_path = option:dep("lc_llvm_path")
    local lc_embree_path = option:dep("lc_embree_path")
    local lc_enable_xir = option:dep("lc_enable_xir")
    local lc_vk_backend_use_xir_spirv = option:dep("lc_vk_backend_use_xir_spirv")
    local lc_vk_backend_use_ast_llvm_spirv = option:dep("lc_vk_backend_use_ast_llvm_spirv")

    -- Enable XIR if fallback backend or LLVM path is set
    if lc_fallback_backend:enabled() or lc_llvm_path:enabled() then
        lc_enable_xir:enable(true, {
            force = true
        })
    end

    -- AST LLVM codegen requires an explicit LLVM installation and XIR, and
    -- disables XIR→SPIR-V.
    if lc_vk_backend_use_ast_llvm_spirv:enabled() then
        local configured_llvm_path = get_config("lc_llvm_path")
        if type(configured_llvm_path) ~= "string" or
            configured_llvm_path:len() == 0 then
            raise("lc_vk_backend_use_ast_llvm_spirv requires " ..
                  "lc_llvm_path to name an LLVM installation built with " ..
                  "the native SPIR-V target.")
        end
        lc_enable_xir:enable(true, {force = true})
        lc_vk_backend_use_xir_spirv:enable(false, {force = true})
    end

    -- Disable XIR→SPIR-V if XIR not enabled
    if not lc_enable_xir:enabled() then
        lc_vk_backend_use_xir_spirv:enable(false, {
            force = false
        })
    end

    -- Disable fallback backend if LLVM or Embree is missing
    if (not lc_llvm_path:enabled()) or (not lc_embree_path:enabled()) then
        lc_fallback_backend:enable(false, {
            force = true
        })
        if lc_fallback_backend:enabled() then
            utils.error("XIR not enabled. Fallback-backend force disabled.")
        end
    end

    -- Platform-specific backend validation

    -- DirectX backend is Windows-only
    local lc_dx_backend = option:dep("lc_dx_backend")
    if lc_dx_backend:enabled() and not is_win then
        lc_dx_backend:enable(false, {
            force = true
        })
        if lc_dx_backend:enabled() then
            utils.error("DX backend not supported in this platform, force disabled.")
        end
    end

    -- Metal backend is macOS-only
    local lc_metal_backend = option:dep("lc_metal_backend")
    if lc_metal_backend:enabled() and not is_plat("macosx") then
        lc_metal_backend:enable(false, {
            force = true
        })
        if lc_metal_backend:enabled() then
            utils.error("Metal backend not supported in this platform, force disabled.")
        end
    end

    -- CUDA backend validation (Windows/Linux only)
    local lc_cuda_ext_lcub = option:dep("lc_cuda_ext_lcub")
    local lc_cuda_backend = option:dep("lc_cuda_backend")
    if lc_cuda_backend:enabled() and not (is_win or is_plat("linux")) then
        lc_cuda_backend:enable(false, {
            force = true
        })
        if lc_cuda_backend:enabled() then
            utils.error("CUDA backend not supported in this platform, force disabled.")
        end
    end

    -- CUDA extensions require CUDA backend
    if lc_cuda_ext_lcub:enabled() and not lc_cuda_backend:enabled() then
        lc_cuda_ext_lcub:enable(false, {
            force = true
        })
        if lc_cuda_ext_lcub:enabled() then
            utils.error("CUDA lcub extension not supported when cuda is disabled")
        end
    end

    -- Enable GUI when Python bindings are enabled
    if enable_py:enabled() then
        option:dep("lc_enable_gui"):enable(true, {
            force = true
        })
    end

    -- Enable DX-CUDA interop when both backends are available
    local lc_dx_cuda_interop = option:dep("lc_dx_cuda_interop")
    if lc_cuda_backend:enabled() and lc_dx_backend:enabled() then
        lc_dx_cuda_interop:enable(true)
    end

    -- Enable VK-CUDA interop when both backends are available
    local lc_vk_cuda_interop = option:dep("lc_vk_cuda_interop")
    if lc_cuda_backend:enabled() and option:dep("lc_vk_backend"):enabled() then
        lc_vk_cuda_interop:enable(true)
    end
end)
option_end()

-- ============================================================================
-- SECTION 2: Build Rules
-- ============================================================================

-- Basic settings rule applied to all targets
rule("lc_basic_settings")
on_config(function(target)
    -- Linux-specific: Use libc++ with Clang
    -- See: https://github.com/LuisaGroup/LuisaCompute/issues/58
    if target:is_plat("linux") then
        if target:has_tool("cxx", "clang", "clangxx") then
            target:add("cxflags", "-stdlib=libc++", {
                force = true
            })
            target:add("syslinks", "c++")
        end
    end
    -- Note: LTO is disabled by default
end)

on_load(function(target)
    -- Helper function to get configuration value from multiple sources
    local function _get_or(name, default_value)
        local v = target:extraconf("rules", "lc_basic_settings", name)
        name = 'lc_' .. name
        if v == nil then
            v = target:values(name)
        end
        if v == nil then
            v = get_config(name)
        end
        if v then
            return v
        end
        return default_value or false
    end

    local function empty_str(value)
        return type(value) == 'string' and #value == 0
    end

    -- Apply toolchain configuration
    local toolchain = _get_or("toolchain")
    if toolchain and not empty_str(toolchain) then
        target:set("toolchains", toolchain)
    end

    -- Apply project type (static/shared library, executable, etc.)
    local project_kind = _get_or("project_kind")
    if project_kind and not empty_str(project_kind) then
        target:set("kind", project_kind)
    end

    -- Linux: Position independent code for static libraries
    if target:is_plat("linux") then
        if project_kind == "static" or project_kind == "object" then
            target:add("cxflags", "-fPIC")
        end
    end

    -- macOS-specific flags
    if target:is_plat("macosx") then
        target:add("cxflags", "-no-pie")
        target:add("cxflags", "-Wno-invalid-specialization", {
            tools = {"clang"}
        })
    end

    -- Enable FMA (Fused Multiply-Add) on x64 platforms
    if target:is_arch("x64", "x86_64") then
        target:add("cxflags", "-mfma", {
            tools = {"clang", "gcc"}
        })
    end

    -- Set C/C++ language standards
    local c_standard = _get_or("c_standard")
    local cxx_standard = _get_or("cxx_standard")
    if c_standard and not empty_str(c_standard) then
        target:set("languages", c_standard, {
            public = true
        })
    end
    if c_standard and not empty_str(cxx_standard) then
        target:set("languages", cxx_standard, {
            public = true
        })
    end

    -- Configure exception handling
    local enable_exception = _get_or("enable_exception")
    if not empty_str(enable_exception) then
        if enable_exception then
            target:set("exceptions", "cxx")
        else
            target:set("exceptions", "no-cxx")
            if target:is_plat('windows') then
                target:add('defines', '_HAS_EXCEPTIONS=0')
            end
        end
    end

    -- Mode-specific configurations
    local win_runtime
    local opt
    if is_mode("debug") then
        win_runtime = _get_or('win_runtime', 'MDd')
        opt = _get_or("optimize", "none")
        target:add("cxflags", "/GS", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    elseif is_mode("releasedbg") then
        win_runtime = _get_or('win_runtime', 'MD')
        opt = _get_or("optimize", "none")
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    else
        win_runtime = _get_or('win_runtime', 'MD')
        opt = _get_or("optimize", "aggressive")
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    end

    if not empty_str(opt) then
        target:set("optimize", opt)
    end

    local warnings = _get_or("warnings", "none")
    if not empty_str(warnings) then
        target:set("warnings", warnings)
    end

    if not empty_str(win_runtime) then
        target:set("runtimes", win_runtime, {
            public = true
        })
    end

    -- MSVC-specific preprocessor settings
    target:add("cxflags", "/Zc:preprocessor", "/wd4244", {
        tools = "cl",
        public = true
    });

    -- SIMD extensions configuration
    if _get_or("enable_simd") then
        if is_arch("arm64") then
            -- NEON is always available on aarch64; vectorexts "neon" emits
            -- -mfpu=neon which is ARM32-only and errors on ARM64 macOS/Linux.
            if not target:is_plat("macosx", "linux") then
                target:add("vectorexts", "neon", {
                    public = true
                })
            end
        else
            target:add("vectorexts", "avx", "avx2", {
                public = true
            })
        end
    end

    -- Link Time Optimization (LTO) configuration
    local use_lto = _get_or("lto", false)
    if not empty_str(use_lto) then
        target:set("policy", "build.optimization.lto", use_lto)
        if use_lto then
            -- Use LLVM tools when using Clang toolchain with LTO
            if toolchain:find("clang") or toolchain:find("llvm") then
                target:set("toolset", "ld", "lld-link")
                target:set("toolset", "ar", "llvm-ar")
            end
        end

    end
    -- RTTI (Run-Time Type Information) configuration
    local use_rtti = _get_or("rtti", false)
    if not empty_str(use_rtti) then
        if use_rtti or has_config("_lc_enable_py") then
            -- Enable RTTI
            target:add("cxflags", "/GR", {
                tools = {"clang_cl", "cl"}
            })
        else
            -- Disable RTTI
            target:add("cxflags", "/GR-", {
                tools = {"clang_cl", "cl"}
            })
            target:add("cxflags", "-fno-rtti", "-fno-rtti-data", {
                tools = {"clang"}
            })
            target:add("cxflags", "-fno-rtti", {
                tools = {"gcc"}
            })
        end
    end
end)
rule_end()

-- Rule to rename extension targets with 'luisa-ext-' prefix
rule("lc-rename-ext")
on_load(function(target)
    target:set("basename", "luisa-ext-" .. target:name())
end)
rule_end()

-- ============================================================================
-- SECTION 3: Validation Targets
-- ============================================================================

-- Windows SDK version checker
-- Ensures Windows SDK 10.0.22000.0 or later is used
target("lc-check-winsdk")
set_kind("phony")
on_config(function(target)
    if not target:is_plat("windows") then
        return nil
    end

    -- Try to get Windows SDK version from toolchain
    local toolchain_settings = target:toolchain("msvc")
    if not toolchain_settings then
        toolchain_settings = target:toolchain("clang-cl")
    end
    if not toolchain_settings then
        toolchain_settings = target:toolchain("llvm")
    end
    if not toolchain_settings then
        return nil
    end

    local sdk_version = toolchain_settings:runenvs().WindowsSDKVersion
    local legal_sdk = false

    if sdk_version then
        import("core.base.semver")
        local ver = semver.match(sdk_version)
        if ver then
            if ver:major() > 10 then
                legal_sdk = true
            elseif ver:major() == 10 and ver:patch() >= 22000 then
                legal_sdk = true
            end
        end

        if not legal_sdk then
            raise("Illegal windows SDK version, requires 10.0.22000.0 or later")
        end
    end
end)
target_end()

-- ============================================================================
-- SECTION 4: SDK Installation Rule
-- ============================================================================

--[[
    SDK Installation Rule - Handles downloading and installing external SDKs
    
    Usage examples:
    
    1. With explicit SDK directory:
    add_rules('lc_install_sdk', {
        sdk_dir = xxx,
        libnames = {{
            name = yyy,
            extract_dir = xxx,    -- Extract to default or target directory
            copy_dir = "",        -- No copy or copy to target directory
            plat_spec = true      -- Transform name to yyyy-linux-x64.zip
        }}
    })
    
    2. Simple library list:
    add_rules('lc_install_sdk', { libnames = xxx })
    
    3. With SDK directory and simple library name:
    add_rules('lc_install_sdk', {
        sdk_dir = xxx,
        libnames = { name = yyy }
    })
--]]
rule('lc_install_sdk')
-- If compile depends on installing, use: set_policy('build.fence', true)
before_build(function(target)
    import("find_sdk")
    find_sdk.on_install_sdk(target, 'lc_install_sdk')
end)
rule_end()

-- ============================================================================
-- SECTION 5: LLVM Integration Rule
-- ============================================================================

-- Rule for linking against LLVM libraries
rule('lc_llvm')
on_load(function(target, opt)
    local libs = {}
    local lc_llvm_path = get_config("lc_llvm_path")
    if not lc_llvm_path then
        return nil
    end

    -- Try build/debug or build/release first, fall back to lc_llvm_path directly
    local llvm_build_dir
    local debug_dir = path.join(lc_llvm_path, "build", "debug")
    local release_dir = path.join(lc_llvm_path, "build", "release")
    if is_mode("debug") and os.exists(debug_dir) then
        llvm_build_dir = debug_dir
    elseif is_mode("release") and os.exists(release_dir) then
        llvm_build_dir = release_dir
    else
        llvm_build_dir = lc_llvm_path
    end

    -- Add LLVM library and include paths
    target:add("linkdirs", path.join(llvm_build_dir, "lib"))
    target:add("includedirs", path.join(llvm_build_dir, "include"))
    -- Also add LLVM source include dir (for in-tree builds where headers are split)
    local llvm_src_include = path.join(lc_llvm_path, "llvm", "include")
    if os.exists(llvm_src_include) then
        target:add("includedirs", llvm_src_include)
    end
    local clang_src_include = path.join(lc_llvm_path, "clang", "include")
    if os.exists(clang_src_include) then
        target:add("includedirs", clang_src_include)
    end

    -- Collect all LLVM libraries (excluding LLVM-C)
    for __, filepath in ipairs(os.files(path.join(llvm_build_dir, "lib/*.lib"))) do
        local basename = path.basename(filepath)
        if basename:match("LLVM") ~= nil and basename ~= "LLVM-C" then
            table.insert(libs, basename)
        end
    end
    target:add("links", libs)

    -- Platform-specific system libraries
    if is_plat("windows") then
        target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", 'Ws2_32', 'ntdll')
    elseif is_plat("linux") then
        target:add("syslinks", "uuid")
    elseif is_plat("macosx") then
        target:add("frameworks", "CoreFoundation")
    end
end)

-- Copy LLVM DLLs to output directory on Windows
after_build(function(target)
    import("async.jobgraph")
    import("async.runjobs")

    if not is_plat("windows") then
        return nil
    end

    local lc_llvm_path = get_config("lc_llvm_path")
    if not lc_llvm_path then
        return nil
    end

    -- Use same build dir resolution as on_load
    local llvm_build_dir
    local debug_dir = path.join(lc_llvm_path, "build", "debug")
    local release_dir = path.join(lc_llvm_path, "build", "release")
    if is_mode("debug") and os.exists(debug_dir) then
        llvm_build_dir = debug_dir
    elseif is_mode("release") and os.exists(release_dir) then
        llvm_build_dir = release_dir
    else
        llvm_build_dir = lc_llvm_path
    end

    local function copy(src_path, dst_path)
        os.cp(src_path, dst_path, {
            copy_if_different = true,
            async = true,
            detach = true
        })
    end

    local dst_path = target:targetdir()
    local jobs = jobgraph.new()

    for __, filepath in ipairs(os.files(path.join(llvm_build_dir, "bin/*.dll"))) do
        jobs:add(filepath, function()
            copy(filepath, path.join(dst_path, path.filename(filepath)))
        end)
    end

    runjobs("copy_llvm_job", jobs, {
        comax = 1000,
        timeout = -1,
        timer = function(running_jobs_indices)
            utils.error("timeout.")
        end
    })
end)
rule_end()

-- ============================================================================
-- SECTION 7: Target Execution Rule
-- ============================================================================

-- Rule for running built targets with proper working directory
rule("lc_run_target")
on_run(function(target)
    import("core.base.option")

    -- Get target name from rule config or use target name
    local name = target:extraconf("rules", "lc_run_target", "name")
    if not name then
        name = target:name()
    end

    local arguments = option.get("arguments")
    local tar_dir = path.absolute(target:targetdir())

    os.execv(path.join(tar_dir, name), arguments, {
        curdir = tar_dir
    })
end)
rule_end()

-- ============================================================================
-- SECTION 8: Global Configuration Functions
-- ============================================================================

--[[
    Global project configuration helper function.
    
    This function is used across the project to apply consistent build settings.
    It handles:
    - Unity build configuration (when enabled)
    - Basic settings rule application
    
    For submodule usage, existing rules can be preserved by defining _config_rules
    before this file is loaded.
--]]

-- Initialize default config rules if not already set (for submodule support)
if _config_rules == nil then
    _config_rules = {"lc_basic_settings"}
end

-- Unity build configuration
if _disable_unity_build == nil then
    local unity_build = get_config("lc_enable_unity_build")
    if unity_build ~= nil then
        _disable_unity_build = not unity_build
    end
end

-- Main project configuration function
if not _config_project then
    function _config_project(config)
        -- Apply unity build if enabled and batch size is valid
        local batch_size = config["batch_size"]
        if type(batch_size) == "number" and batch_size > 1 and (not _disable_unity_build) then
            add_rules("c.unity_build", {
                batchsize = batch_size
            })
            add_rules("c++.unity_build", {
                batchsize = batch_size
            })
        end

        -- Apply configuration rules
        if type(_config_rules) == "table" then
            add_rules(_config_rules, config)
        end
    end
end
