table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("volk", "stb")
target("btree")
set_kind("headeronly")
add_includedirs("BTree/include", {public = true})
target_end()
-- ext
lc_eastl_enable_custom_malloc = has_config("lc_enable_custom_malloc")
lc_eastl_enable_mimalloc = has_config("lc_enable_mimalloc")
includes("EASTL")
-- Every Vulkan build compiles its backend-private kernels to SPIR-V at build
-- time. Native user-shader codegen is optional, but glslang and SPIRV-Tools
-- are not: the builtins must never fall back to runtime DXC compilation.
local need_vulkan_spirv_tools = has_config("lc_vk_backend")
if need_vulkan_spirv_tools then
    includes("glslang")
    target("lc-glslang")
    add_defines("ENABLE_HLSL", {public = true})
    add_files("glslang/glslang/HLSL/*.cpp")
    target_end()
end

if not has_config("lc_spdlog_use_xrepo") then
    includes("spdlog")
end
if not has_config("lc_reproc_use_xrepo") then
    includes("reproc")
end
if not has_config("lc_lmdb_use_xrepo") then
    includes("liblmdb")
end
lc_eastl_enable_mimalloc = nil
lc_eastl_enable_custom_malloc = nil
-- yyjson
if not has_config("lc_yyjson_use_xrepo") then
    target("lc-yyjson")
    _config_project({
        project_kind = "static"
    })
    on_load(function(target)
        local src_path = path.join(os.scriptdir(), "yyjson/src")
        target:add("files", path.join(src_path, "yyjson.c"))
        target:add("includedirs", src_path, {
            public = true
        })
        target:add("cxflags", "/utf-8", {
            tools = "cl"
        })
    end)
    target_end()
end
-- The HLSL validation test compiles DXC output to SPIR-V and validates it with
-- SPIRV-Tools even when neither of the optional Vulkan SPIR-V code generators
-- is enabled. Keep that test dependency aligned with src/tests/xmake.lua.
local need_spv_tools = need_vulkan_spirv_tools or
                       (has_config("lc_enable_tests") and
                        (has_config("lc_vk_backend") or
                         has_config("lc_dx_backend")))
if need_spv_tools then
    target('spirv-headers')
    set_kind('headeronly')
    add_includedirs("spirv-headers/include", "spirv-headers/include/spirv/unified1", {
        public = true
    })
    target_end()

    includes("SPIRV-Tools")
    -- SPIRV-Tools' broad source/*.cpp glob also picks up the optional
    -- mimalloc override, whose header is intentionally unavailable when the
    -- project allocator is disabled. CMake only compiles this source when
    -- SPIRV_TOOLS_USE_MIMALLOC is enabled; mirror that default here.
    target("spirv-tools")
    remove_files("SPIRV-Tools/source/mimalloc.cpp")
    target_end()
end

-- ============================================================================
-- TVM (apache/tvm submodule): xmake port of the upstream CMake build.
--
-- Upstream split the core into the tirx/ subproject (apache-tvm TIRx); the
-- parent tvm/ tree keeps the extension sources. The three xmake libraries
-- the TileIR -> TVM TIRx bridge links against fold the upstream targets:
--   * tvm_ffi      (tirx/3rdparty/tvm-ffi: object model, containers, extra CXX API)
--   * tvm_runtime  (tirx core runtime + parent VM/RPC extensions)
--   * tvm_compiler (tirx core compiler + codegen + parent te/s_tir/topi/relax)
-- The source lists below mirror the file(GLOB) calls in tirx/CMakeLists.txt
-- and tvm/CMakeLists.txt (GLOB -> "*.cc", GLOB_RECURSE -> "**.cc") with the
-- default options (no CUDA/ROCm/Vulkan/Metal device runtimes, RPC + threads
-- on, optional LLVM via lc_tvm_llvm).
-- Only configured when lc_tile_tirx_bridge is enabled and the submodule is
-- checked out.
-- ============================================================================
local lc_tvm_root = path.join(os.scriptdir(), "tvm")
local lc_tirx_root = path.join(lc_tvm_root, "tirx")
local lc_tvm_enabled = has_config("lc_tile_tirx_bridge") or has_config("lc_tvm_stack")
if lc_tvm_enabled and os.exists(path.join(lc_tvm_root, "CMakeLists.txt")) then
    -- Optional LLVM codegen for tvm_compiler (mirrors
    -- tirx/cmake/modules/LLVM.cmake with TVM_TIRX_USE_LLVM=ON). Resolved from
    -- --lc_llvm_path (an LLVM installation with include/ and lib/) or, failing
    -- that, the xmake-repo llvm package.
    local tvm_use_llvm = has_config("lc_tvm_llvm")
    local tvm_llvm_path = get_config("lc_llvm_path")
    if tvm_use_llvm and (type(tvm_llvm_path) ~= "string" or tvm_llvm_path == "") then
        tvm_llvm_path = nil
        add_requires("llvm", {system = false})
    end
    local tvm_ffi_root = path.join(lc_tirx_root, "3rdparty/tvm-ffi")
    -- Public include interface shared by every TVM library and its consumers:
    -- core tvm/*.h in tirx/include, parent extension headers (relax, s_tir,
    -- te, topi, runtime/vm, runtime/disco) in tvm/include, plus tvm/ffi/*.h
    -- and dlpack/dlpack.h.
    local tvm_public_includes = {
        path.join(lc_tirx_root, "include"),
        path.join(lc_tvm_root, "include"),
        path.join(tvm_ffi_root, "include"),
        path.join(tvm_ffi_root, "3rdparty/dlpack", "include")
    }
    -- Common cross-platform settings for all TVM libraries (mirrors the global
    -- flags in tirx/CMakeLists.txt and tirx/3rdparty/tvm-ffi/CMakeLists.txt).
    -- The exports macros are defined privately so the produced DLL exports its
    -- symbols on MSVC (TVM_FFI_EXPORTS / TVM_RUNTIME_EXPORTS /
    -- TVM_RUNTIME_EXTRA_EXPORTS / TVM_TIRX_EXPORTS / TVM_EXPORTS).
    local function tvm_common_config(...)
        set_languages("cxx20")
        set_warnings("none")
        add_defines("TVM_INDEX_DEFAULT_I64=1", {public = true})
        add_defines(...)
        add_includedirs(tvm_public_includes, {public = true})
        if is_plat("windows") then
            -- These targets skip _config_project, so apply the project's
            -- lc_win_runtime override here (see scripts/options.lua).
            local runtime = get_config("lc_win_runtime")
            if type(runtime) ~= "string" or runtime == "" then
                runtime = is_mode("debug") and "MDd" or "MD"
            end
            set_runtimes(runtime)
        end
        -- compiler-rt builtins (half-precision) are included as SYSTEM headers
        add_sysincludedirs(path.join(lc_tirx_root, "3rdparty", "compiler-rt"))
        if is_plat("windows") then
            add_cxflags("/EHsc", "/MP", "/bigobj", {tools = {"cl", "clang_cl"}})
            -- TVM uses std::aligned_storage with extended alignment (MSVC
            -- requires opting into the conforming behaviour).
            add_defines("_ENABLE_EXTENDED_ALIGNED_STORAGE")
            -- tvm-ffi walks stack traces through DbgHelp on Windows
            -- (libbacktrace is disabled, TVM_FFI_USE_LIBBACKTRACE=0 below).
            add_syslinks("DbgHelp")
        else
            -- HIDE_PRIVATE_SYMBOLS=ON in the CMake build; exported symbols are
            -- marked with visibility("default") by the TVM_*_DLL macros.
            add_cxflags("-fvisibility=hidden", "-fvisibility-inlines-hidden")
            add_syslinks("pthread", "dl")
            if is_plat("macosx") then
                add_rpathdirs("@loader_path")
            else
                add_rpathdirs("$ORIGIN")
            end
        end
    end

    -- tvm_ffi: C++ object model and FFI (tirx/3rdparty/tvm-ffi).
    target("tvm_ffi")
    set_kind("shared")
    tvm_common_config("TVM_FFI_EXPORTS")
    -- libbacktrace is intentionally disabled to keep the xmake build
    -- self-contained and cross-platform (matches TVM_FFI_USE_LIBBACKTRACE=OFF).
    add_defines("TVM_FFI_USE_LIBBACKTRACE=0", "TVM_FFI_BACKTRACE_ON_SEGFAULT=0")
    -- Endianness is baked in by CMake; every supported platform here is
    -- little-endian.
    add_defines("TVM_FFI_CMAKE_LITTLE_ENDIAN=1", {public = true})
    add_files(path.join(tvm_ffi_root, "src/ffi/*.cc"))
    -- TVM_FFI_USE_EXTRA_CXX_API=ON (CMake default): reflection, structural
    -- equal/hash, serialization and module loading used by tvm_compiler.
    add_files(path.join(tvm_ffi_root, "src/ffi/extra/*.cc"))
    target_end()

    -- tvm_runtime: tirx core runtime (RUNTIME_SRCS in tirx/CMakeLists.txt)
    -- plus the parent VM/RPC extensions (tvm_runtime_extra in
    -- tvm/CMakeLists.txt; there is no src/runtime/memory in current TVM).
    target("tvm_runtime")
    set_kind("shared")
    add_deps("tvm_ffi")
    tvm_common_config("TVM_RUNTIME_EXPORTS", "TVM_RUNTIME_EXTRA_EXPORTS", "TVM_FFI_EXPORTS")
    add_files(path.join(lc_tirx_root, "src/runtime/*.cc"),
              path.join(lc_tvm_root, "src/runtime/vm/*.cc"),
              path.join(lc_tvm_root, "src/runtime/rpc/*.cc"))
    target_end()
    -- tvm_compiler: tirx core compiler stack (COMPILER_SRCS + CODEGEN_SRCS in
    -- tirx/CMakeLists.txt) plus the parent compiler extensions (COMPILER_SRCS
    -- in tvm/CMakeLists.txt).
    target("tvm_compiler")
    set_kind("shared")
    add_deps("tvm_runtime")
    tvm_common_config("TVM_EXPORTS", "TVM_TIRX_EXPORTS", "TVM_FFI_EXPORTS")
    -- tvm/CMakeLists.txt bakes the parent package version in globally.
    add_defines("TVM_VERSION=\"0.28.dev0\"")
    add_files(-- tirx core compiler (GLOB_RECURSE in tirx/CMakeLists.txt)
              path.join(lc_tirx_root, "src/ir/**.cc"),
              -- shared symbolic analysis (upstream src/sym; renamed from src/arith).
              path.join(lc_tirx_root, "src/sym/**.cc"),
              path.join(lc_tirx_root, "src/tirx/**.cc"),
              -- src/support is header-only in current TVM (no .cc files).
              path.join(lc_tirx_root, "src/script/**.cc"),
              -- codegen (CPU source codegen + per-backend kind registration)
              path.join(lc_tirx_root, "src/target/*.cc"),
              path.join(lc_tirx_root, "src/codegen/*.cc"),
              path.join(lc_tirx_root, "src/codegen/c/*.cc"),
              path.join(lc_tirx_root, "src/codegen/canonicalizer/*.cc"),
              -- available without LLVM; the LLVM glue is added in on_load below.
              path.join(lc_tirx_root, "src/codegen/llvm/target_kind.cc"),
              path.join(lc_tirx_root, "src/codegen/llvm/canonicalizer/*.cc"),
              path.join(lc_tirx_root, "src/backend/cuda/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/cuda/op/*.cc"),
              path.join(lc_tirx_root, "src/backend/cuda/transforms/*.cc"),
              path.join(lc_tirx_root, "src/backend/cuda/tile/*.cc"),
              path.join(lc_tirx_root, "src/backend/cuda/layout/*.cc"),
              path.join(lc_tirx_root, "src/backend/hexagon/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/metal/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/metal/op/*.cc"),
              path.join(lc_tirx_root, "src/backend/opencl/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/opencl/op/*.cc"),
              path.join(lc_tirx_root, "src/backend/rocm/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/trn/codegen/*.cc"),
              path.join(lc_tirx_root, "src/backend/trn/op/*.cc"),
              path.join(lc_tirx_root, "src/backend/trn/transform/*.cc"),
              path.join(lc_tirx_root, "src/backend/trn/layout/*.cc"),
              path.join(lc_tirx_root, "src/backend/vulkan/codegen/target_kind.cc"),
              path.join(lc_tirx_root, "src/backend/vulkan/codegen/vulkan_fallback_module.cc"),
              path.join(lc_tirx_root, "src/backend/webgpu/codegen/*.cc"),
              -- parent compiler extensions (GLOB_RECURSE in tvm/CMakeLists.txt;
              -- note relax/training is intentionally not part of COMPILER_SRCS)
              path.join(lc_tvm_root, "src/te/**.cc"),
              path.join(lc_tvm_root, "src/s_tir/**.cc"),
              path.join(lc_tvm_root, "src/topi/**.cc"),
              -- relax
              path.join(lc_tvm_root, "src/relax/ir/**.cc"),
              path.join(lc_tvm_root, "src/relax/op/**.cc"),
              path.join(lc_tvm_root, "src/relax/analysis/**.cc"),
              path.join(lc_tvm_root, "src/relax/transform/**.cc"),
              path.join(lc_tvm_root, "src/relax/backend/vm/**.cc"),
              path.join(lc_tvm_root, "src/relax/backend/adreno/**.cc"),
              path.join(lc_tvm_root, "src/relax/backend/task_extraction.cc"),
              path.join(lc_tvm_root, "src/relax/backend/pattern_registry.cc"),
              path.join(lc_tvm_root, "src/relax/utils.cc"),
              path.join(lc_tvm_root, "src/relax/distributed/**.cc"),
              path.join(lc_tvm_root, "src/relax/script/**.cc"),
              path.join(lc_tvm_root, "src/relax/testing/*.cc"),
              -- TVM-side additions owned by this repository (src/ext/tvm_ext)
              -- instead of by the src/ext/tvm submodule. codegen_cuda.cc
              -- ICHECKs tirx.intrinsics.cuda.header_generator / get_codegen,
              -- which upstream only registers from the Python package, so a
              -- pure C++ embedding needs them compiled in here.
              -- Regenerate with scripts/port_cuda_header_generator.py.
              path.join(os.scriptdir(), "tvm_ext", "*.cc"))
    on_load(function(target)
        if not tvm_use_llvm then
            return
        end
        -- LLVM 22's Support library references RtlGetLastNtStatus from
        -- ntdll.dll on Windows.
        if is_plat("windows") then
            target:add("syslinks", "ntdll")
        end
        -- COMPILER_LLVM_SRCS from tirx/cmake/modules/LLVM.cmake
        -- (target_kind.cc is already part of the base CODEGEN_SRCS above).
        target:add("files", path.join(lc_tirx_root, "src/codegen/llvm/*.cc"),
                   path.join(lc_tirx_root, "src/backend/cuda/codegen/llvm/*.cc"),
                   path.join(lc_tirx_root, "src/backend/rocm/codegen/llvm/*.cc"),
                   path.join(lc_tirx_root, "src/backend/hexagon/codegen/llvm/*.cc"))
        target:remove("files", path.join(lc_tirx_root, "src/codegen/llvm/target_kind.cc"))
        -- LLVM.cmake builds the glue with NDEBUG (LLVM debug-symbol link issues).
        target:add("defines", "NDEBUG=1")
        if not is_plat("windows") then
            -- upstream builds the LLVM glue without RTTI to match the
            -- canonical (RTTI-off) LLVM binaries
            target:add("cxxflags", "-fno-rtti")
        end
        local function llvm_version_def(version_str)
            local major = tonumber(version_str:match("^(%d+)")) or 0
            local minor = tonumber(version_str:match("^%d+%.(%d+)")) or 0
            if major < 15 then
                raise("lc_tvm_llvm requires LLVM 15 or newer (found " .. version_str .. ").")
            end
            return major * 10 + minor
        end
        if tvm_llvm_path then
            local include_dir = path.join(tvm_llvm_path, "include")
            local lib_dir = path.join(tvm_llvm_path, "lib")
            target:add("sysincludedirs", include_dir)
            target:add("linkdirs", lib_dir)
            local config_h = path.join(include_dir, "llvm", "Config", "llvm-config.h")
            local content = os.exists(config_h) and io.readfile(config_h) or ""
            local major = tonumber(content:match("#define%s+LLVM_VERSION_MAJOR%s+(%d+)")) or 0
            local minor = tonumber(content:match("#define%s+LLVM_VERSION_MINOR%s+(%d+)")) or 0
            if major < 15 then
                raise("lc_tvm_llvm requires LLVM 15 or newer (found " ..
                      major .. "." .. minor .. " in " .. config_h .. ").")
            end
            target:add("defines", "TVM_LLVM_VERSION=" .. (major * 10 + minor))
            -- Link every LLVM component library, mirroring the fallback
            -- backend; a monolithic libLLVM is linked alone when present.
            local libs = {}
            local pattern = is_plat("windows") and "*.lib" or "libLLVM*.*"
            local mono = is_plat("windows") and nil or os.files(path.join(lib_dir, "libLLVM.*"))
            if mono and #mono > 0 then
                table.insert(libs, "LLVM")
            else
                for _, filepath in ipairs(os.files(path.join(lib_dir, pattern))) do
                    local basename = path.basename(filepath)
                    if basename:match("^LLVM") or basename:match("^libLLVM") then
                        local name = basename:gsub("^lib", "")
                        if name ~= "LLVM-C" and not name:match("DLL$") then
                            table.insert(libs, name)
                        end
                    end
                end
            end
            target:add("links", libs)
            local has_aarch64 = false
            for _, lib in ipairs(libs) do
                if lib:match("AArch64") then has_aarch64 = true break end
            end
            if not has_aarch64 then
                has_aarch64 = os.exists(path.join(include_dir, "llvm", "Config", "Targets.def"))
            end
            target:add("defines", "TVM_LLVM_HAS_AARCH64_TARGET=" .. (has_aarch64 and 1 or 0))
        else
            target:add("packages", "llvm")
            -- Detect the version from the installed headers; the package
            -- instance does not expose a version accessor in this context.
            local pkg = target:pkg("llvm")
            local version = "0.0"
            if pkg then
                local config_h = path.join(pkg:installdir(), "include", "llvm", "Config", "llvm-config.h")
                local content = os.exists(config_h) and io.readfile(config_h) or ""
                local major = content:match("#define%s+LLVM_VERSION_MAJOR%s+(%d+)")
                local minor = content:match("#define%s+LLVM_VERSION_MINOR%s+(%d+)")
                if major then version = major .. "." .. (minor or "0") end
            end
            target:add("defines", "TVM_LLVM_VERSION=" .. llvm_version_def(version))
            -- xmake-repo llvm builds all targets; conservatively enable AArch64
            target:add("defines", "TVM_LLVM_HAS_AARCH64_TARGET=1")
        end
    end)
    target_end()
end

table.remove(_config_rules, rename_rule_idx)
