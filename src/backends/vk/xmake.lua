local vk_xmake_dir = os.scriptdir()

-- Host-side compiler used only while building Vulkan backend-private kernels.
-- The generated intrinsic header stays in xmake's autogen directory rather
-- than modifying the bundled glslang submodule.
target("lc-glslang-standalone")
set_basename("luisa-glslang")
set_kind("binary")
set_default(false)
set_policy("build.fence", true)
_config_project({
    project_kind = "binary"
})
add_deps("lc-glslang")
add_files("../../ext/glslang/StandAlone/StandAlone.cpp")
on_load(function(target)
    target:add("includedirs", target:autogendir())
    if target:is_plat("windows") then
        target:add("syslinks", "psapi")
    end
end)
before_build(function(target)
    import("core.project.depend")
    import("lib.detect.find_tool")
    local glslang_dir = path.normalize(path.join(
        vk_xmake_dir, "../../ext/glslang"))
    local generated_header = path.join(
        target:autogendir(), "glslang", "glsl_intrinsic_header.h")
    local generator = path.join(glslang_dir, "gen_extension_headers.py")
    local input_dir = path.join(
        glslang_dir, "glslang", "ExtensionHeaders")
    local dependencies = os.files(path.join(input_dir, "*"))
    table.insert(dependencies, generator)
    depend.on_changed(function()
        local python = find_tool("python3") or find_tool("python")
        assert(python, "Python is required to generate glslang intrinsic headers.")
        os.mkdir(path.directory(generated_header))
        os.vrunv(python.program, {
            generator, "-i", input_dir, "-o", generated_header
        })
    end, {
        dependfile = target:dependfile(generated_header),
        files = dependencies,
        changed = target:is_rebuilt() or not os.isfile(generated_header)
    })
end)
target_end()

target("lc-vk-validate-spirv")
set_basename("luisa-validate-spirv")
set_kind("binary")
set_default(false)
set_policy("build.fence", true)
_config_project({
    project_kind = "binary"
})
add_deps("spirv-tools")
add_files("../common/spirv/validate_spirv.cpp")
target_end()

target("lc-vk-embed-device-lib")
set_basename("luisa-embed-device-lib")
set_kind("binary")
set_default(false)
set_policy("build.fence", true)
_config_project({
    project_kind = "binary"
})
add_files("../../../utils/embed_device_lib.cpp")
target_end()

target("lc-backend-vk")
set_basename("luisa-backend-vk")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-runtime", "lc-vstl", "lc-hlsl-codegen",
         "lc-glslang-standalone", "lc-vk-validate-spirv",
         "lc-vk-embed-device-lib")
add_headerfiles("*.h")
add_files("*.cpp")
lc_set_pcxxheader("lc_vk_pch.h")

on_load(function(target)
    if not has_config('lc_vk_backend_enable_dxc_compatibility') and
       not has_config('lc_vk_backend_use_xir_spirv') and
       not has_config('lc_vk_backend_use_ast_llvm_spirv') then
        raise("The Vulkan backend needs a native SPIR-V codegen route " ..
              "when DXC compatibility is disabled.")
    end
    local generated_dir = path.join(target:autogendir(), "vk_builtin")
    target:add("includedirs", generated_dir)
    target:add("files", path.join(
        generated_dir, "vulkan_builtin_spirv_embedded.cpp"), {
            always_added = true
        })
    if target:is_plat("windows") then
        target:add("defines", "NOMINMAX", "VK_USE_PLATFORM_WIN32_KHR")
    elseif target:is_plat("linux") then
        target:add("defines", "VK_USE_PLATFORM_XCB_KHR", "VK_USE_PLATFORM_XLIB_KHR")
    end
    local function rela(p)
        return path.normalize(path.join(os.scriptdir(), p))
    end
    target:add("headerfiles", rela("../common/default_binary_io.h"))
    target:add("files", rela("../common/default_binary_io.cpp"))
    target:add("deps", "lc-volk")
    if target:is_plat("macosx") then
        target:add("files", rela("../common/moltenvk_surface.mm"))
        target:add("frameworks", "Foundation", "Metal", "QuartzCore", "AppKit")
    end
    if has_config("lc_vk_cuda_interop") then
        target:add("defines", "LUISA_VULKAN_ENABLE_CUDA_INTEROP")
        target:add("links", "nvrtc_static", "cudart_static", "cuda")
        target:add('deps', '_lc_cuda_base')
    end
    if has_config('lc_vk_backend_use_ast_llvm_spirv')  or has_config('lc_vk_backend_use_xir_spirv') then
        target:add('deps', 'lc-spirv')
    end
end)
before_build(function(target)
    import("core.project.depend")

    local generated_dir = path.join(target:autogendir(), "vk_builtin")
    local embedded_cpp = path.join(
        generated_dir, "vulkan_builtin_spirv_embedded.cpp")
    local embedded_h = path.join(
        generated_dir, "vulkan_builtin_spirv_embedded.h")
    local common_dir = path.normalize(path.join(vk_xmake_dir, "../common"))
    local builtin_dir = path.join(vk_xmake_dir, "builtin")
    local sources = {
        path.join(builtin_dir, "indirect_prepare.comp.hlsl"),
        path.join(builtin_dir, "accel_process.comp.hlsl"),
        path.join(builtin_dir, "bindless_upload.comp.hlsl")
    }
    local outputs = {
        path.join(generated_dir, "indirect_prepare.spv"),
        path.join(generated_dir, "accel_process.spv"),
        path.join(generated_dir, "bindless_upload.spv")
    }
    local dependencies = {
        path.join(common_dir, "indirect_dispatch_layout.def"),
        path.join(builtin_dir, "vulkan_accel_update_layout.def"),
        path.join(vk_xmake_dir, "xmake.lua")
    }
    for _, source in ipairs(sources) do
        table.insert(dependencies, source)
    end

    local glslang = target:dep("lc-glslang-standalone"):targetfile()
    local validator = target:dep("lc-vk-validate-spirv"):targetfile()
    local embedder = target:dep("lc-vk-embed-device-lib"):targetfile()
    table.insert(dependencies, glslang)
    table.insert(dependencies, validator)
    table.insert(dependencies, embedder)

    local missing_output = not os.isfile(embedded_cpp) or
                           not os.isfile(embedded_h)
    for _, output in ipairs(outputs) do
        missing_output = missing_output or not os.isfile(output)
    end
    depend.on_changed(function()
        os.mkdir(generated_dir)
        for index, source in ipairs(sources) do
            os.vrunv(glslang, {
                "-D", "-V", "--target-env", "vulkan1.2",
                "-S", "comp", "-e", "main", "-I" .. common_dir,
                "-o", outputs[index], source
            })
            os.vrunv(validator, {outputs[index]})
        end
        local embed_arguments = {}
        for _, output in ipairs(outputs) do
            table.insert(embed_arguments, output)
        end
        for _, argument in ipairs({
            "--unsigned", "--preserve-ext",
            "--prefix", "luisa_compute_vk_builtin_",
            "-o", embedded_cpp, "-h", embedded_h
        }) do
            table.insert(embed_arguments, argument)
        end
        os.vrunv(embedder, embed_arguments)
    end, {
        dependfile = target:dependfile(embedded_cpp),
        files = dependencies,
        changed = target:is_rebuilt() or missing_output
    })
end)
if has_config('lc_vk_backend_use_xir_spirv') then
    add_defines('LUISA_XIR_TO_SPIRV')
end
-- AST LLVM → SPIR-V codegen path
if has_config('lc_vk_backend_use_ast_llvm_spirv') then
    add_deps('lc-spirv-llvm')
    add_defines('LUISA_AST_LLVM_TO_SPIRV')
end
if not has_config('lc_vk_backend_enable_dxc_compatibility') then
    add_defines('LC_NO_HLSL_BUILTIN')
end
target_end()
