#include "shader_compiler.h"
#include "../env_flag.h"
#include "../spirv_fma_rewrite.h"
#include <atomic>
#include <fstream>
#include <luisa/core/dynamic_module.h>
#include <luisa/vstl/string_utility.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/spin_mutex.h>
#include <luisa/core/stl/filesystem.h>
#ifndef _WIN32
#define WINAPI
#endif
namespace lc::hlsl {
#ifndef LC_DXC_THROW_IF_FAILED
#define LC_DXC_THROW_IF_FAILED(x)                  \
    {                                              \
        HRESULT hr_ = (x);                         \
        LUISA_ASSERT(hr_ == S_OK, "bad HRESULT."); \
    }
#endif
namespace {

// Resolves `#include`d files against the raw include spelling first (absolute
// paths and cwd-relative names) and then against each registered include
// directory, in order. Plain reference-counted IUnknown: the handler lives on
// the stack for the synchronous `IDxcCompiler3::Compile` call and is never
// deleted through the interface.
class FileSystemIncludeHandler final : public IDxcIncludeHandler {
    IDxcUtils *_utils;
    luisa::span<const luisa::filesystem::path> _include_dirs;
    std::atomic<ULONG> _ref_count{1};

    [[nodiscard]] bool try_load(
        const luisa::filesystem::path &path,
        IDxcBlob **ppIncludeSource) const {
        // The ate-open doubles as the existence test and the size probe.
        std::ifstream file{path, std::ios::in | std::ios::binary | std::ios::ate};
        if (!file.is_open()) return false;
        auto end = file.tellg();
        if (end < 0) return false;
        luisa::string content;
        content.resize(static_cast<size_t>(end));
        file.seekg(0, std::ios::beg);
        if (!content.empty() &&
            !file.read(content.data(), static_cast<std::streamsize>(content.size()))) {
            return false;
        }
        // CreateBlob copies the content, so the local buffer may die here.
        IDxcBlobEncoding *blob{nullptr};
        if (FAILED(_utils->CreateBlob(
                content.empty() ? "" : content.data(), (UINT32)content.size(),
                DXC_CP_ACP, &blob))) {
            return false;
        }
        *ppIncludeSource = blob;
        return true;
    }

public:
    FileSystemIncludeHandler(
        IDxcUtils *utils,
        luisa::span<const luisa::filesystem::path> include_dirs) noexcept
        : _utils{utils}, _include_dirs{include_dirs} {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) override {
        if (ppvObject == nullptr) return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDxcIncludeHandler)) {
            *ppvObject = static_cast<IDxcIncludeHandler *>(this);
            AddRef();
            return S_OK;
        }
        *ppvObject = nullptr;
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return _ref_count.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    ULONG STDMETHODCALLTYPE Release() override {
        return _ref_count.fetch_sub(1, std::memory_order_relaxed) - 1;
    }

    HRESULT STDMETHODCALLTYPE LoadSource(
        _In_z_ LPCWSTR pFilename,
        _COM_Outptr_result_maybenull_ IDxcBlob **ppIncludeSource) override {
        *ppIncludeSource = nullptr;
        // Absolute paths and cwd-relative spellings resolve as-is; the open
        // inside try_load doubles as the existence test.
        if (try_load(pFilename, ppIncludeSource)) return S_OK;
        for (auto &&dir : _include_dirs) {
            auto candidate = dir / pFilename;
            if (try_load(candidate, ppIncludeSource)) return S_OK;
        }
        return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    }
};

}// namespace

static vstd::wstring GetSM(uint shaderModel) {
    vstd::string smStr;
    smStr << vstd::to_string(shaderModel / 10) << '_' << vstd::to_string(shaderModel % 10);
    vstd::wstring wstr;
    wstr.resize(smStr.size());
    for (size_t i = 0; i < smStr.size(); ++i) {
        wstr[i] = smStr[i];
    }
    return wstr;
}
IDxcCompiler3 *ShaderCompiler::compiler() const {
    return compiler_module.comp;
}
IDxcUtils *ShaderCompiler::utils() const {
    return compiler_module.utils;
}
IDxcLibrary *ShaderCompiler::library() const {
    return compiler_module.library;
}
ShaderCompiler::~ShaderCompiler() = default;
ShaderCompilerModule::ShaderCompilerModule(luisa::filesystem::path const &path, bool is_spirv)
    : comp{nullptr},
      library{nullptr},
      utils{nullptr},
      dxil([&]() {
          if (!is_spirv) return luisa::DynamicModule::load(path, "dxil");
          return luisa::DynamicModule{};
      }()),
      dxcCompiler(luisa::DynamicModule::load(path, "dxcompiler")) {
    if (!is_spirv && !dxil) {
        LUISA_ERROR_WITH_LOCATION("dxil.dll not found.");
    }
    if (!dxcCompiler) {
        LUISA_ERROR("dxcompiler.dll not found.");
    }
    auto voidPtr = dxcCompiler.address("DxcCreateInstance");
    auto DxcCreateInstance =
        reinterpret_cast<HRESULT(WINAPI *)(const IID &, const IID &, LPVOID *)>(voidPtr);
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&comp));
    DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library));
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
}
ShaderCompilerModule::~ShaderCompilerModule() {
    // TODO: directx-compiler may crash here
    utils->Release();
    library->Release();
    comp->Release();
}
ShaderCompiler::ShaderCompiler(luisa::filesystem::path const &path, bool is_spirv)
    : compiler_module(path, is_spirv) {
}
CompileResult ShaderCompiler::compile(
    vstd::string_view code,
    vstd::span<LPCWSTR> args,
    IDxcIncludeHandler *include_handler) const {
    DxcBuffer buffer{
        code.data(),
        code.size(),
        CP_ACP};
    ComUniquePtr<IDxcResult> compileResult;
    auto comp = compiler();
    if (comp) {
        IDxcResult *ptr{};
        auto compile_result = comp->Compile(
            &buffer,
            args.data(),
            args.size(),
            include_handler,
            IID_PPV_ARGS(&ptr));
        LC_DXC_THROW_IF_FAILED(compile_result);
        compileResult = ComUniquePtr<IDxcResult>{ptr};
    }
    HRESULT status;
    LC_DXC_THROW_IF_FAILED(compileResult->GetStatus(&status));
    if (status == 0) {
        ComUniquePtr<IDxcBlob> resultBlob;
        IDxcBlob *resultBlobPtr{};
        LC_DXC_THROW_IF_FAILED(compileResult->GetResult(&resultBlobPtr));
        resultBlob = ComUniquePtr<IDxcBlob>{resultBlobPtr};
        return resultBlob;
    } else {
        ComUniquePtr<IDxcBlobEncoding> errBuffer;
        IDxcBlobEncoding *errBufferPtr{};
        LC_DXC_THROW_IF_FAILED(compileResult->GetErrorBuffer(&errBufferPtr));
        errBuffer = ComUniquePtr<IDxcBlobEncoding>{errBufferPtr};
        auto errStr = vstd::string_view(
            reinterpret_cast<char const *>(errBuffer->GetBufferPointer()),
            errBuffer->GetBufferSize());
        return vstd::string(errStr);
    }
}
template<typename Vec>
static void AddCompileFlags(Vec &args, bool debug, bool /*spirv*/) {
    vstd::push_back_all(
        args,
        {DXC_ARG_ALL_RESOURCES_BOUND,
         L"-enable-16bit-types",
         DXC_ARG_PACK_MATRIX_ROW_MAJOR,
         L"-HV 2021"});
    if (debug) {
        args.emplace_back(DXC_ARG_DEBUG);
    } else {
        args.emplace_back(L"-no-warnings");
    }
}
// Shader-capability extensions the HLSL-to-SPIR-V route may use
// (`hlsl::SpirvShaderCapabilities`).  dxc rejects an `-fspv-extension=` name it does
// not know, so a capability is only enabled when the device negotiated it; the
// generated source then contains the matching instruction, whose OpCapability and
// OpExtension dxc declares on its own (see builtin/hlsl_header.bytes).
//
// `-fspv-extension=` is an allow-list, not an addition: as soon as one name is given,
// dxc drops every extension the option does not name - *including* the ones that are
// enabled by default (`FeatureManager::allowAllKnownExtensions` resets every
// non-default extension, and the option path resets everything but the named ones).
// The generated source relies on parts of that default set: an inline ray query needs
// `SPV_KHR_ray_query`, a non-uniform descriptor index (every bindless access) needs
// `SPV_EXT_descriptor_indexing`.  Naming them here whenever a capability is enabled is
// therefore not optional - without it a shader that compiled before the capability
// became available would start failing with "extension ... not permitted to use".
// `KHR` is dxc's spelling for "every KHR extension", and permitting an extension never
// emits anything by itself, so the module still declares only what its instructions
// require.
//
// VK_KHR_maintenance9 is the one capability that is *not* an extension name: its
// bitwise relaxation is selected by the dedicated `-fspv-enable-maintenance9`
// switch, which only changes the operand width of instructions that already exist.
// It still has to participate in the early return below, because it needs the same
// allow-list preservation as the extension-named capabilities.
template<typename Vec>
static void AddSpirvCapabilityFlags(Vec &args, SpirvShaderCapabilities capabilities) {
    // Do NOT pass -fspv-extension=SPV_KHR_fma or the blanket -fspv-extension=KHR
    // (which includes SPV_KHR_fma): DXC's bundled SPIRV-Tools omits "SPV_KHR_fma"
    // from the extension allowlists of AggressiveDCEPass, LocalAccessChainConvertPass,
    // LocalSingleBlockLoadStoreElimPass, and LocalSingleStoreElimPass, causing all
    // four legalization/optimization passes to silently no-op and leaving invalid
    // storage-buffer function variables alive. Instead, DXC lowers `mad()` to core
    // GLSL.std.450 Fma so spirv-opt legalizes and optimizes fully, and
    // promote_glsl_fma_to_op_fma_khr upgrades float32 Fma to OpFmaKHR afterwards.
    if (!capabilities.abort &&
        !capabilities.cluster_acceleration_structure &&
        !capabilities.maintenance9_bitwise &&
        !capabilities.mixed_float_dot_product &&
        !capabilities.split_barrier) {
        return;// no allow-list at all, so dxc keeps its default set (minus SPV_KHR_fma)
    }
    args.emplace_back(L"-fspv-extension=SPV_KHR_8bit_storage");
    args.emplace_back(L"-fspv-extension=SPV_KHR_16bit_storage");
    args.emplace_back(L"-fspv-extension=SPV_KHR_device_group");
    args.emplace_back(L"-fspv-extension=SPV_KHR_fragment_shading_rate");
    args.emplace_back(L"-fspv-extension=SPV_KHR_non_semantic_info");
    args.emplace_back(L"-fspv-extension=SPV_KHR_multiview");
    args.emplace_back(L"-fspv-extension=SPV_KHR_shader_draw_parameters");
    args.emplace_back(L"-fspv-extension=SPV_KHR_post_depth_coverage");
    args.emplace_back(L"-fspv-extension=SPV_KHR_ray_tracing");
    args.emplace_back(L"-fspv-extension=SPV_KHR_shader_clock");
    args.emplace_back(L"-fspv-extension=SPV_KHR_ray_query");
    args.emplace_back(L"-fspv-extension=SPV_KHR_physical_storage_buffer");
    args.emplace_back(L"-fspv-extension=SPV_KHR_vulkan_memory_model");
    args.emplace_back(L"-fspv-extension=SPV_KHR_compute_shader_derivatives");
    args.emplace_back(L"-fspv-extension=SPV_KHR_fragment_shader_barycentric");
    args.emplace_back(L"-fspv-extension=SPV_KHR_maximal_reconvergence");
    args.emplace_back(L"-fspv-extension=SPV_KHR_float_controls");
    args.emplace_back(L"-fspv-extension=SPV_KHR_quad_control");
    args.emplace_back(L"-fspv-extension=SPV_KHR_workgroup_memory_explicit_layout");
    args.emplace_back(L"-fspv-extension=SPV_KHR_bfloat16");
    args.emplace_back(L"-fspv-extension=SPV_KHR_opacity_micromap");
    args.emplace_back(L"-fspv-extension=SPV_KHR_constant_data");
    args.emplace_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");
    args.emplace_back(L"-fspv-extension=SPV_EXT_shader_atomic_float_add");
    args.emplace_back(L"-fspv-extension=SPV_EXT_shader_atomic_float_min_max");
    args.emplace_back(L"-fspv-extension=SPV_EXT_demote_to_helper_invocation");
    args.emplace_back(L"-fspv-extension=SPV_EXT_shader_image_int64");
    args.emplace_back(L"-fspv-extension=SPV_EXT_long_vector");
    args.emplace_back(L"-fspv-extension=SPV_EXT_float8");
    args.emplace_back(L"-fspv-extension=SPV_NV_cooperative_vector");
    args.emplace_back(L"-fspv-extension=SPV_NV_ray_tracing_motion_blur");
    args.emplace_back(L"-fspv-extension=SPV_EXT_shader_invocation_reorder");
    args.emplace_back(L"-fspv-extension=SPV_GOOGLE_hlsl_functionality1");
    args.emplace_back(L"-fspv-extension=SPV_GOOGLE_user_type");
    if (capabilities.abort) {
        args.emplace_back(L"-fspv-extension=SPV_KHR_abort");
    }
    if (capabilities.maintenance9_bitwise) {
        args.emplace_back(L"-fspv-enable-maintenance9");
    }
    if (capabilities.cluster_acceleration_structure) {
        args.emplace_back(L"-fspv-extension=SPV_NV_cluster_acceleration_structure");
    }
    if (capabilities.mixed_float_dot_product) {
        args.emplace_back(L"-fspv-extension=SPV_VALVE_mixed_float_dot_product");
    }
    if (capabilities.split_barrier) {
        args.emplace_back(L"-fspv-extension=SPV_EXT_split_barrier");
    }
}

template<typename Vec>
static void AddUnsafeMathFlags(Vec &args, bool spirv) {
    if (spirv) {
        args.emplace_back(L"-ffinite-math-only");
    }
}
CompileResult ShaderCompiler::_compile_with_profile(
    vstd::string_view code,
    vstd::wstring_view profile,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug,
    vstd::string_view entry_point,
    luisa::span<const luisa::filesystem::path> include_dirs,
    SpirvShaderCapabilities spirv_capabilities,
    bool spirv_fcgl) const {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 64> args;
    // DXC takes the entry point / target as NUL-terminated wide strings, so the
    // two that are built here (rather than being literals) need an owning local
    // that outlives the argument vector below.
    vstd::wstring smStr{profile};
    vstd::wstring entry_str;
    if (!entry_point.empty()) {
        entry_str.resize(entry_point.size());
        for (size_t i = 0; i < entry_point.size(); ++i) {
            entry_str[i] = static_cast<wchar_t>(entry_point[i]);
        }
    }
    if (spirv) {
        args.emplace_back(L"-spirv");
        args.emplace_back(L"/DSPV");
        if (shaderModel > 65) {
            args.emplace_back(L"-fspv-target-env=vulkan1.3");
        } else if (shaderModel > 60) {
            args.emplace_back(L"-fspv-target-env=vulkan1.2");
        }
        AddSpirvCapabilityFlags(args, spirv_capabilities);
    }
    if (!entry_str.empty()) {
        args.emplace_back(L"-E");
        args.emplace_back(entry_str.c_str());
    }
    args.emplace_back(L"-T");
    args.emplace_back(smStr.c_str());
    AddCompileFlags(args, debug, spirv);
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args, spirv);
    }
    if (spirv && spirv_fcgl) {
        // Emit unlegalized SPIR-V directly so the caller can run SPIRV-Tools
        // legalization and optimization with interleaved compact-ids and
        // bounded SROA/loop-unroll.
        args.emplace_back(L"-fcgl");
    } else if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    auto postprocess_spirv_fma = [&](CompileResult result) -> CompileResult {
        if (!spirv || spirv_fcgl || !spirv_capabilities.fma ||
            !result.is_type_of<ComUniquePtr<IDxcBlob>>()) {
            return result;
        }
        auto &blob = result.get<0>();
        const size_t byte_size = blob->GetBufferSize();
        if (byte_size < 20u || (byte_size % sizeof(uint32_t)) != 0u) {
            return result;
        }
        std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
        std::memcpy(words.data(), blob->GetBufferPointer(), byte_size);
        if (luisa::compute::detail::promote_glsl_fma_to_op_fma_khr(words) == 0u) {
            return result;
        }
        IDxcBlobEncoding *rewritten_ptr{};
        LC_DXC_THROW_IF_FAILED(utils()->CreateBlob(
            words.data(),
            static_cast<UINT32>(words.size() * sizeof(uint32_t)),
            CP_ACP,
            &rewritten_ptr));
        return ComUniquePtr<IDxcBlob>{rewritten_ptr};
    };
    if (include_dirs.empty()) {
        // Historical behaviour: no include handler is registered at all.
        return postprocess_spirv_fma(compile(code, args));
    }
    // Stack-lifetime handler; the synchronous Compile call below never
    // outlives it. The handler searches the directories, so no `-I` args.
    FileSystemIncludeHandler handler{utils(), include_dirs};
    return postprocess_spirv_fma(compile(code, args, &handler));
}
CompileResult ShaderCompiler::compile_compute(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug,
    vstd::string_view entry_point,
    luisa::span<const luisa::filesystem::path> include_dirs,
    SpirvShaderCapabilities spirv_capabilities,
    bool spirv_fcgl) const {
    vstd::wstring profile;
    profile << L"cs_" << GetSM(shaderModel);
    return _compile_with_profile(
        code, profile, optimize, shaderModel, enableUnsafeMath, spirv, debug,
        entry_point, include_dirs, spirv_capabilities, spirv_fcgl);
}
CompileResult ShaderCompiler::compile_graphics_stage(
    vstd::string_view code,
    bool vertex,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug,
    vstd::string_view entry_point,
    luisa::span<const luisa::filesystem::path> include_dirs,
    SpirvShaderCapabilities spirv_capabilities,
    bool spirv_fcgl) const {
    vstd::wstring profile;
    profile << (vertex ? L"vs_" : L"ps_") << GetSM(shaderModel);
    return _compile_with_profile(
        code, profile, optimize, shaderModel, enableUnsafeMath, spirv, debug,
        entry_point, include_dirs, spirv_capabilities, spirv_fcgl);
}
RasterBin ShaderCompiler::compile_raster(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug,
    SpirvShaderCapabilities spirv_capabilities) const {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 64> args;
    AddCompileFlags(args, debug, spirv);
    if (spirv) {
        args.emplace_back(L"/DSPV");
        args.emplace_back(L"-spirv");
        if (shaderModel > 65) {
            args.emplace_back(L"-fspv-target-env=vulkan1.3");
        } else if (shaderModel > 60) {
            args.emplace_back(L"-fspv-target-env=vulkan1.2");
        }
        AddSpirvCapabilityFlags(args, spirv_capabilities);
    }
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args, spirv);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    args.emplace_back(L"-T");
    auto size = args.size();
    vstd::wstring smStr;
    smStr << L"vs_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DVS");
    auto postprocess_spirv_fma = [&](CompileResult result) -> CompileResult {
        if (!spirv || !spirv_capabilities.fma ||
            !result.is_type_of<ComUniquePtr<IDxcBlob>>()) {
            return result;
        }
        auto &blob = result.get<0>();
        const size_t byte_size = blob->GetBufferSize();
        if (byte_size < 20u || (byte_size % sizeof(uint32_t)) != 0u) {
            return result;
        }
        std::vector<uint32_t> words(byte_size / sizeof(uint32_t));
        std::memcpy(words.data(), blob->GetBufferPointer(), byte_size);
        if (luisa::compute::detail::promote_glsl_fma_to_op_fma_khr(words) == 0u) {
            return result;
        }
        IDxcBlobEncoding *rewritten_ptr{};
        LC_DXC_THROW_IF_FAILED(utils()->CreateBlob(
            words.data(),
            static_cast<UINT32>(words.size() * sizeof(uint32_t)),
            CP_ACP,
            &rewritten_ptr));
        return ComUniquePtr<IDxcBlob>{rewritten_ptr};
    };
    RasterBin bin;
    bin.vertex = postprocess_spirv_fma(compile(code, args));
    luisa::vector_resize(args, size);
    smStr.clear();
    smStr << L"ps_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DPS");
    bin.pixel = postprocess_spirv_fma(compile(code, args));
    return bin;
}
CompileResult ShaderCompiler::compile_raytracing(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv,
    bool debug,
    SpirvShaderCapabilities spirv_capabilities) const {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 64> args;
    vstd::wstring smStr;
    smStr << L"lib_" << GetSM(shaderModel);
    if (spirv) {
        args.emplace_back(L"-spirv");
        args.emplace_back(L"/DSPV");
        if (shaderModel > 60) {
            args.emplace_back(L"-fspv-target-env=vulkan1.2");
        }
        // Enable ray tracing SPIR-V extensions
        args.emplace_back(L"-fspv-extension=SPV_KHR_ray_tracing");
        AddSpirvCapabilityFlags(args, spirv_capabilities);
    }
    args.emplace_back(L"-T");
    args.emplace_back(smStr.c_str());
    AddCompileFlags(args, debug, spirv);
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args, spirv);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    return compile(code, args);
}
#undef LC_DXC_THROW_IF_FAILED
}// namespace lc::hlsl
