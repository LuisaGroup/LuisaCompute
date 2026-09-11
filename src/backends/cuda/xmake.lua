if has_config("lc_cuda_ext_lcub") then
    includes("lcub")
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
    if has_config("lc_reproc_use_xrepo") then
        target:add("packages", "reproc")
    else
        target:add("deps", "reproc")
    end
    if has_config("lc_cuda_ext_lcub") then
        target:add("deps", "lc-compute-cuda-ext-lcub")
    end
    target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
    local src_path = os.scriptdir()
    local exclude_files = {}
    exclude_files["cuda_nvrtc_compiler.cpp"] = true
    exclude_files["cuda_builtin_embedded.cpp"] = true
    exclude_files["cuda_devrt_embedded.cpp"] = true
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
    if has_config('lc_llvm_path') then
        target:add("defines", 'LUISA_ENABLE_XIR', 'LUISA_COMPUTE_ENABLE_LLVM')
        target:add("files", path.join(os.scriptdir(), 'llvm_codegen/*.cpp'))
    end
    target:add("defines", "LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
    target:add("deps", "lc-vulkan-swapchain", "lc-volk")
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

target("lc-nvrtc")
_config_project({
    project_kind = "binary",
    runtime = "MT"
})
if is_plat("windows") then
    add_syslinks("Ws2_32", "User32")
end
set_basename("luisa_nvrtc")
add_deps("lc-cuda-backend-base")
add_links("nvrtc_static", "nvrtc-builtins_static", "nvptxcompiler_static")
add_files("cuda_nvrtc_compiler.cpp")
target_end()
