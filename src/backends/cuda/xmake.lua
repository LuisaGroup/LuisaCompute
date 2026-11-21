if has_config("lc_cuda_ext_lcub") then
    includes("lcub")
end

-- ============================================================================
-- CUDA 13.4+ native Tile IR
-- ============================================================================
-- tile/cuda_tile_ir.cpp needs two toolkit artifacts that only CUDA 13.4+ ships:
-- the standalone NVRTC helper in this directory (it compiles with --enable-tile
-- and therefore needs the nvrtcGetTileIR* entry points) and the `tileiras`
-- assembler. The CMake build (src/backends/cuda/CMakeLists.txt) builds that
-- helper only when the configured toolkit is new enough; this is the same gate
-- for xmake. Both resolve the toolkit the way the rest of the backend does
-- (cuda_sdkdir() -> find_cuda(), the one _lc_cuda_base publishes to every
-- target), and the version is read from the toolkit's own headers instead of
-- being guessed from a configuration value.
--
-- `readfile` is io.readfile of the calling target script: project scripts have
-- neither `io` nor `import`, so reading the headers stays in the target scope.
local function _lc_cuda_tile_ir_sdk(cuda, readfile)
    if type(cuda) ~= "table" or type(readfile) ~= "function" then
        return nil
    end
    -- The native Tile toolchain is a Windows/Linux part of the toolkit, exactly
    -- like the CMake gate.
    if not (is_plat("windows") or is_plat("linux")) then
        return nil
    end
    local includedirs = cuda.includedirs
    local include_dir = type(includedirs) == "table" and includedirs[1] or includedirs
    if type(include_dir) ~= "string" or include_dir == "" then
        return nil
    end
    if not os.isfile(path.join(include_dir, "cuda_tile.h")) then
        return nil
    end
    -- Version evidence: CUDA_VERSION of the toolkit's own cuda.h (13050 = 13.5).
    local cuda_h = path.join(include_dir, "cuda.h")
    local cuda_h_text = os.isfile(cuda_h) and readfile(cuda_h) or ""
    local version = tonumber(tostring(cuda_h_text):match("#%s*define%s+CUDA_VERSION%s+(%d+)"))
    if not version or version < 13040 then
        return nil
    end
    -- The helper is compiled against nvrtcGetTileIR*/nvrtcGetTileIRSize: an
    -- include directory from another toolkit has to fail here, not at link time.
    local nvrtc_h = path.join(include_dir, "nvrtc.h")
    local nvrtc_h_text = os.isfile(nvrtc_h) and readfile(nvrtc_h) or ""
    if not tostring(nvrtc_h_text):find("nvrtcGetTileIR", 1, true) then
        return nil
    end
    if type(cuda.bindir) ~= "string" or cuda.bindir == "" then
        return nil
    end
    local assembler = path.join(cuda.bindir, is_plat("windows") and "tileiras.exe" or "tileiras")
    if not os.isfile(assembler) then
        return nil
    end
    -- Both values become C string literals: forward slashes and escaped quotes,
    -- exactly like LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR below.
    local include_text = include_dir:gsub("\\", "/"):gsub('"', '\\"')
    local assembler_text = assembler:gsub("\\", "/"):gsub('"', '\\"')
    return {
        version = version,
        include_dir = include_text,
        assembler = assembler_text
    }
end

target("lc-cuda-backend-base")
set_kind("phony")
add_deps('_lc_cuda_base')
on_load(function(target)
    if target:is_plat("windows") then
        target:add("defines", "UNICODE", "_CRT_SECURE_NO_WARNINGS", {
            public = true
        })
        target:add("syslinks", "Cfgmgr32", "Advapi32", {
            public = true
        })
    else
        -- The CUDA toolchain only contributes $CUDA_PATH/lib, while the Linux
        -- toolkit packages put the static device/runtime libraries
        -- (libcudadevrt.a, libcudart_static.a, libnvrtc_static.a) in lib64.
        -- Publish both directories so the -l flags actually resolve.
        local cuda_home = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or "/usr/local/cuda"
        for _, subdir in ipairs({"lib64", "lib"}) do
            local lib_dir = path.join(cuda_home, subdir)
            if os.isdir(lib_dir) then
                target:add("linkdirs", lib_dir, {public = true, force = true})
            end
        end
    end
end)
target_end()

target('lc-cuda-backend-builtin')
set_kind('static')
add_rules("utils.bin2obj", {
    extensions = {".cu", ".h"}
})
add_files('cuda_builtin/*.cu', 'cuda_builtin/*.h')
target_end()

target("lc-backend-cuda")
set_basename("luisa-backend-cuda")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-runtime", "lc-cuda-backend-base", 'lc-cuda-backend-builtin', "lc-tile")

add_rules('lc_llvm')
lc_set_pcxxheader("lc_cuda_pch.h")
add_headerfiles("*.h")
add_headerfiles("tile/*.h")
add_defines('LUISA_BIN_2_OBJ')
on_load(function(target)
    -- The software ray-tracing fallback (src/backends/common/rtx): the library
    -- that answers `create_mesh` / `create_accel` when the user asked for the
    -- fallback or when the machine has no OptiX (see cuda_config_ext.h and
    -- cuda_device.cpp).  `src/backends/common/xmake.lua` builds it whenever a
    -- backend that can fall back is enabled.
    target:add("deps", "luisa-fallback-rtx")
    if has_config("lc_reproc_use_xrepo") then
        target:add("packages", "reproc")
    else
        target:add("deps", "reproc")
    end
    if has_config("lc_cuda_ext_lcub") then
        target:add("deps", "lc-compute-cuda-ext-lcub")
    end
    target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
    -- Official host headers only; do not add them to the NVRTC/device source bundle.
    local optix_include = path.join(os.scriptdir(), "../../ext/optix/include")
    assert(os.isfile(path.join(optix_include, "optix_function_table.h")),
           "OptiX headers are missing. Run git submodule update --init src/ext/optix")
    target:add("includedirs", optix_include, {system = true})
    local src_path = os.scriptdir()
    local exclude_files = {}
    exclude_files["cuda_nvrtc_compiler.cpp"] = true
    -- The CUDA 13.4 native Tile compiler is its own executable (target
    -- lc-cuda-tile-compiler below): the backend only ever runs it, and without
    -- LUISA_CUDA_TILE_IR_ENABLED the native Tile path keeps reporting itself as
    -- unavailable and never references nvrtcGetTileIR.
    exclude_files["cuda_tile_ir_compiler.cpp"] = true
    exclude_files["cuda_builtin_embedded.cpp"] = true
    if os.isfile(path.join(src_path, "cuda_devrt_embedded.cpp")) then
        target:add("defines", "LUISA_COMPUTE_ENABLE_CUDADEVRT=1")
    else
        exclude_files["cuda_devrt_embedded.cpp"] = true
    end
    exclude_files["cuda_texture_compression.cpp"] = true
    local file_paths = {'*.cpp', 'extensions/*.cpp', 'tile/*.cpp'}
    for _, f in ipairs(file_paths) do
        for _, filepath in ipairs(os.files(path.join(src_path, f))) do
            local file_name = path.filename(filepath)
            if not exclude_files[file_name] then
                target:add("files", filepath)
            end
        end
    end
    -- `--lc_llvm_path=` (empty) is Lua-truthy, so test the value, not the key:
    -- otherwise the LLVM codegen sources are compiled without any LLVM headers.
    local lc_llvm_path = get_config("lc_llvm_path")
    if type(lc_llvm_path) == "string" and lc_llvm_path ~= "" then
        -- XIR kernels, including autodiff, lower directly to LLVM/PTX.
        target:add("defines", 'LUISA_ENABLE_XIR', 'LUISA_COMPUTE_ENABLE_LLVM')
        -- OptiX IR requires CMake's opt-in LLVM 7 writer; xmake remains PTX-only.
        target:add("files", path.join(os.scriptdir(), 'llvm_codegen/*.cpp|cuda_codegen_llvm_optix_ir*.cpp'))
    end
    target:add("defines", "LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
    target:add("deps", "lc-vulkan-swapchain", "lc-volk")
    -- Ordinary CUDA Tile realizers also need these headers without TIRx.
    import("cuda_sdkdir", {rootdir = get_config("lc_scripts_path")})
    import("detect.sdks.find_cuda")
    local cuda = find_cuda(cuda_sdkdir())
    local include_dirs = cuda and cuda.includedirs
    local include_dir = type(include_dirs) == "table" and include_dirs[1] or include_dirs
    assert(type(include_dir) == "string" and include_dir ~= "",
           "CUDA Tile requires the CUDA toolkit include directory")
    include_dir = include_dir:gsub("\\", "/"):gsub('"', '\\"')
    target:add("defines", 'LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR="' .. include_dir .. '"')
    -- Native Tile IR: the same gate as the standalone compiler below. Without
    -- it tile/cuda_tile_ir.cpp keeps the fail-closed "unavailable" answer and
    -- links no NVRTC Tile symbol, so pre-13.4 toolkits behave as before.
    local tile_ir = _lc_cuda_tile_ir_sdk(cuda, io.readfile)
    if tile_ir then
        target:add("defines", "LUISA_CUDA_TILE_IR_ENABLED=1",
                   'LUISA_CUDA_TILE_IR_ASSEMBLER="' .. tile_ir.assembler .. '"')
        -- The helper runs next to the backend binary, so build it with it. It is
        -- an executable, not a library: nothing is inherited or linked (the same
        -- wiring the gallery uses for luisa_nvrtc).
        target:add("deps", "lc-cuda-tile-compiler", {inherit = false, links = false})
    end
      if has_config("lc_tile_tirx_bridge") then
          -- The TIRx bridge sources are compiled into lc-tile for xmake. This
          -- define mirrors the CMake `if (TARGET ...tirx)` wiring in
          -- src/backends/cuda/CMakeLists.txt.
          target:add("defines", "LUISA_CUDA_TILE_TIRX")
          -- cuda_tile.cpp includes the TIRx compiler header, which pulls in
          -- tvm-ffi headers that use throw.
          target:set("exceptions", "cxx")
      end
end)
add_files("extensions/cuda_denoiser.cpp", "extensions/cuda_dstorage.cpp", "extensions/cuda_pinned_memory.cpp")
add_links("cuda")

-- after_build(function(target)
--     import("lib.detect.find_file")
--     import("detect.sdks.find_cuda")
--     import("cuda_sdkdir")
--     local cuda = find_cuda(cuda_sdkdir())
--     if cuda then
--         local linkdirs = cuda["linkdirs"]
--         local lc_bin_dir = target:targetdir()
--         if is_plat("windows") then
--             for i, v in ipairs(linkdirs) do
--                 os.cp(path.join(v, "cudadevrt.lib"), lc_bin_dir)
--             end
--         end
--         -- TODO: linux
--     end
-- end)
target_end()

-- Standalone native Tile compiler (mirror of luisa-cuda-tile-compiler in
-- src/backends/cuda/CMakeLists.txt). It is a helper process, never a library:
-- the backend keeps its ordinary NVRTC path untouched and only runs this
-- executable (see tile/cuda_tile_ir.cpp). The target exists in every
-- configuration so a parent project can depend on the name unconditionally;
-- without a CUDA 13.4+ Tile toolkit it degrades to a phony target that builds
-- nothing, which is what keeps older toolkits fail-closed.
-- It is produced in the ordinary target directory, i.e. next to
-- luisa-backend-cuda and luisa_nvrtc, where the backend looks for it.
target("lc-cuda-tile-compiler")
set_basename("luisa_cuda_tile_compiler")
_config_project({
    project_kind = "binary",
    -- The static NVRTC/nvPTX archives of CUDA 13.4+ are built against the static
    -- CRT, so the helper has to use /MT or link.exe rejects them (LNK2038). This
    -- is the key lc_basic_settings reads; CMake sets the same with
    -- MSVC_RUNTIME_LIBRARY MultiThreaded.
    win_runtime = "MT"
})
on_load(function(target)
    import("cuda_sdkdir", {rootdir = get_config("lc_scripts_path")})
    import("detect.sdks.find_cuda")
    local tile_ir = _lc_cuda_tile_ir_sdk(find_cuda(cuda_sdkdir()), io.readfile)
    if not tile_ir then
        -- No native Tile toolchain in this SDK: nothing to build and nothing to
        -- depend on. The backend reports the feature as unavailable.
        target:set("kind", "phony")
        return
    end
    target:add("deps", "lc-cuda-backend-base")
    target:add("files", path.join(os.scriptdir(), "cuda_tile_ir_compiler.cpp"))
    -- The helper's own protocol is unchanged by the rest of the backend: static
    -- NVRTC plus the archives it references, i.e. what CMake's CUDA::nvrtc_static
    -- interface target links transitively.
    target:add("links", "nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static")
    target:add("defines", 'LUISA_CUDA_TILE_IR_INCLUDE_DIR="' .. tile_ir.include_dir .. '"')
    if target:is_plat("windows", "mingw") then
        target:add("syslinks", "Ws2_32", "ntdll")
        -- Large Tile expressions exceed NVRTC's default 1 MiB thread stack:
        -- reserve more address space without raising the initial commit.
        if target:is_plat("mingw") then
            target:add("ldflags", "-Wl,--stack,16777216", {force = true})
            -- wmain() takes the UTF-16 command line, like CMake's -municode.
            target:add("ldflags", "-municode", {force = true})
        else
            target:add("ldflags", "/STACK:16777216", {force = true})
        end
    end
end)
target_end()

target("lc-nvrtc")
_config_project({
    project_kind = "binary",
    runtime = "MT"
})
if is_plat("windows") then
      add_syslinks("Ws2_32", "User32", "ntdll")
  end
  set_basename("luisa_nvrtc")
add_deps("lc-cuda-backend-base")
add_links("nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static")
add_files("cuda_nvrtc_compiler.cpp")
target_end()
