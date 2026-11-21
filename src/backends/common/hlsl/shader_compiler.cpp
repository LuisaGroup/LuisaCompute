#include "shader_compiler.h"
#include "../env_flag.h"
#include "../spirv_fma_rewrite.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <unordered_map>
#include <luisa/core/dynamic_module.h>
#include <luisa/vstl/string_utility.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/spin_mutex.h>
#include <luisa/core/stl/filesystem.h>
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
    dxc_create_instance = reinterpret_cast<CreateInstanceFn>(voidPtr);
    if (!is_spirv && dxil) {
        dxil_create_instance = reinterpret_cast<CreateInstanceFn>(dxil.address("DxcCreateInstance"));
    }
    dxc_create_instance(CLSID_DxcCompiler, IID_PPV_ARGS(&comp));
    dxc_create_instance(CLSID_DxcLibrary, IID_PPV_ARGS(&library));
    dxc_create_instance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
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
namespace {

[[nodiscard]] size_t large_dxil_fast_opt_threshold() noexcept {
    if (luisa::compute::detail::env_flag("LUISA_DX_DISABLE_FAST_OPT")) {
        return SIZE_MAX;
    }
    if (auto env = luisa::get_environment_variable("LUISA_DX_FAST_OPT_THRESHOLD")) {
        char *end = nullptr;
        auto parsed = std::strtoull(env->c_str(), &end, 10);
        if (end != env->c_str() && *end == '\0') {
            return static_cast<size_t>(parsed);
        }
    }
    // Shaders >= 256 KB benefit strongly from pre-inline SSA promotion and
    // replacing DXC's superlinear `-jump-threading` + redundant `-loop-unroll`
    // passes; smaller kernels compile in milliseconds on the direct path.
    return 256u * 1024u;
}

[[nodiscard]] std::vector<std::wstring> parse_and_rewrite_dxil_o3_passes(
    std::string_view odump_text) {
    std::vector<std::wstring> passes;
    passes.reserve(140u);
    size_t pos = 0u;
    while (pos < odump_text.size()) {
        size_t end = odump_text.find('\n', pos);
        if (end == std::string_view::npos) {
            end = odump_text.size();
        }
        std::string_view line = odump_text.substr(pos, end - pos);
        pos = end + 1u;
        while (!line.empty() &&
               (line.front() == ' ' || line.front() == '\t' ||
                line.front() == '\r' || line.front() == '\0')) {
            line.remove_prefix(1u);
        }
        while (!line.empty() &&
               (line.back() == ' ' || line.back() == '\t' ||
                line.back() == '\r' || line.back() == '\0')) {
            line.remove_suffix(1u);
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (line == "-opt-mod-passes") {
            // Promote callee locals to SSA once before `-always-inline` copies
            // callees into the entry point hundreds of times.
            passes.emplace_back(L"-mem2reg");
            passes.emplace_back(L"-dce");
            passes.emplace_back(L"-opt-mod-passes");
            continue;
        }
        if (line == "-jump-threading") {
            // DXC LLVM 3.7 JumpThreading scales superlinearly (~O(N^2.8)) on
            // large dispatch CFGs. Replace with linear-time constant/CFG cleanup.
            passes.emplace_back(L"-sccp");
            passes.emplace_back(L"-simplifycfg");
            passes.emplace_back(L"-dce");
            continue;
        }
        if (line.starts_with("-loop-unroll")) {
            // Keep `-dxil-loop-unroll` (which honors explicit HLSL [unroll] and
            // bounded loops) while skipping the 3 extra generic LLVM 3.7
            // `-loop-unroll` passes that scale superlinearly on large kernels.
            continue;
        }
        std::wstring wline;
        wline.reserve(line.size());
        for (char c : line) {
            wline.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
        }
        if (wline.starts_with(L"-always-inline,")) {
            auto lt_pos = wline.find(L"InsertLifetime=t");
            if (lt_pos != std::wstring::npos) {
                wline.replace(lt_pos, 16u, L"InsertLifetime=f");
            }
        }
        passes.emplace_back(std::move(wline));
    }
    return passes;
}

[[nodiscard]] uint32_t parse_entry_wave_size(std::string_view code) noexcept {
    constexpr std::string_view kPrefix = "[WaveSize(";
    const size_t pos = code.rfind(kPrefix);
    if (pos == std::string_view::npos) {
        return 0u;
    }
    size_t i = pos + kPrefix.size();
    uint32_t val = 0u;
    size_t digits = 0u;
    while (i < code.size() && code[i] >= '0' && code[i] <= '9') {
        val = val * 10u + static_cast<uint32_t>(code[i] - '0');
        ++i;
        ++digits;
    }
    if (digits == 0u || i + 1u >= code.size() || code[i] != ')' || code[i + 1u] != ']') {
        return 0u;
    }
    return val;
}

struct WaveSizePatchResult {
    bool ok{false};
    bool already_present{false};
    std::string patched_ir;
};

[[nodiscard]] WaveSizePatchResult inject_dxil_entry_wave_size(
    std::string_view ir, uint32_t wave_size) {
    WaveSizePatchResult res;
    constexpr std::string_view kEntryDecl = "\n!dx.entryPoints = !{!";
    const size_t ep_decl = ir.rfind(kEntryDecl);
    if (ep_decl == std::string_view::npos) {
        return res;
    }
    size_t id_pos = ep_decl + kEntryDecl.size();
    size_t id_end = id_pos;
    while (id_end < ir.size() && ir[id_end] >= '0' && ir[id_end] <= '9') {
        ++id_end;
    }
    if (id_end == id_pos || id_end >= ir.size() || ir[id_end] != '}') {
        return res;
    }
    const std::string_view ep_id_str = ir.substr(id_pos, id_end - id_pos);

    // Scan all numbered metadata definitions after !dx.entryPoints to find max_md_id
    // and the entry-point tuple definition `!<ep_id> = !{...}`.
    uint64_t max_md_id = 0u;
    size_t ep_line_start = std::string_view::npos;
    size_t ep_line_end = std::string_view::npos;
    size_t scan = ep_decl + kEntryDecl.size();
    while (scan < ir.size()) {
        const size_t nl = ir.find("\n!", scan);
        if (nl == std::string_view::npos) {
            break;
        }
        size_t p = nl + 2u;
        size_t q = p;
        while (q < ir.size() && ir[q] >= '0' && ir[q] <= '9') {
            ++q;
        }
        if (q > p && q + 4u <= ir.size() && ir.substr(q, 4u) == " = !") {
            uint64_t num = 0u;
            for (size_t k = p; k < q; ++k) {
                num = num * 10u + static_cast<uint64_t>(ir[k] - '0');
            }
            if (num > max_md_id) {
                max_md_id = num;
            }
            if (ir.substr(p, q - p) == ep_id_str) {
                ep_line_start = nl + 1u;
                ep_line_end = ir.find('\n', ep_line_start);
                if (ep_line_end == std::string_view::npos) {
                    ep_line_end = ir.size();
                }
            }
        }
        scan = nl + 2u;
    }
    if (ep_line_start == std::string_view::npos) {
        return res;
    }
    const std::string_view ep_line = ir.substr(ep_line_start, ep_line_end - ep_line_start);
    const size_t ep_close = ep_line.rfind('}');
    const size_t ep_last_comma = ep_line.rfind(", ");
    if (ep_close == std::string_view::npos ||
        ep_last_comma == std::string_view::npos ||
        ep_last_comma + 2u >= ep_close) {
        return res;
    }
    const std::string_view props_tok = ep_line.substr(ep_last_comma + 2u, ep_close - (ep_last_comma + 2u));

    if (props_tok == "null") {
        const uint64_t props_id = max_md_id + 1u;
        const uint64_t wave_id = max_md_id + 2u;
        const size_t replace_pos = ep_line_start + ep_last_comma + 2u;
        res.patched_ir.reserve(ir.size() + 80u);
        res.patched_ir.append(ir.substr(0u, replace_pos));
        res.patched_ir.append("!");
        res.patched_ir.append(std::to_string(props_id));
        res.patched_ir.append(ir.substr(ep_line_start + ep_close));
        if (!res.patched_ir.empty() && res.patched_ir.back() != '\n') {
            res.patched_ir.push_back('\n');
        }
        res.patched_ir.append("!");
        res.patched_ir.append(std::to_string(props_id));
        res.patched_ir.append(" = !{i32 11, !");
        res.patched_ir.append(std::to_string(wave_id));
        res.patched_ir.append("}\n!");
        res.patched_ir.append(std::to_string(wave_id));
        res.patched_ir.append(" = !{i32 ");
        res.patched_ir.append(std::to_string(wave_size));
        res.patched_ir.append("}\n");
        res.ok = true;
        return res;
    }

    if (props_tok.size() < 2u || props_tok[0] != '!') {
        return res;
    }
    std::string props_needle;
    props_needle.reserve(props_tok.size() + 6u);
    props_needle.push_back('\n');
    props_needle.append(props_tok);
    props_needle.append(" = !{");
    const size_t props_pos = ir.find(props_needle, ep_decl);
    if (props_pos == std::string_view::npos) {
        return res;
    }
    const size_t props_line_start = props_pos + 1u;
    size_t props_line_end = ir.find('\n', props_line_start);
    if (props_line_end == std::string_view::npos) {
        props_line_end = ir.size();
    }
    const std::string_view props_line = ir.substr(props_line_start, props_line_end - props_line_start);
    // Tag 11 is kDxilWaveSizeTag; tag 23 is kDxilWaveSizeRangeTag.
    if (props_line.find("i32 11,") != std::string_view::npos ||
        props_line.find("i32 23,") != std::string_view::npos) {
        res.ok = true;
        res.already_present = true;
        return res;
    }
    const size_t props_close = props_line.rfind('}');
    if (props_close == std::string_view::npos) {
        return res;
    }
    const uint64_t wave_id = max_md_id + 1u;
    const size_t insert_pos = props_line_start + props_close;
    res.patched_ir.reserve(ir.size() + 64u);
    res.patched_ir.append(ir.substr(0u, insert_pos));
    res.patched_ir.append(", i32 11, !");
    res.patched_ir.append(std::to_string(wave_id));
    res.patched_ir.append(ir.substr(insert_pos));
    if (!res.patched_ir.empty() && res.patched_ir.back() != '\n') {
        res.patched_ir.push_back('\n');
    }
    res.patched_ir.append("!");
    res.patched_ir.append(std::to_string(wave_id));
    res.patched_ir.append(" = !{i32 ");
    res.patched_ir.append(std::to_string(wave_size));
    res.patched_ir.append("}\n");
    res.ok = true;
    return res;
}

struct WaveSizeBitcodePatchResult {
    bool ok{false};
    bool already_present{false};
    std::vector<uint8_t> patched_bc;
};

struct BitcodeReader {
    const uint8_t *data{nullptr};
    size_t size{0u};
    uint64_t bitpos{0u};
    bool error{false};

    [[nodiscard]] uint64_t total_bits() const noexcept {
        return static_cast<uint64_t>(size) * 8u;
    }

    uint64_t read(uint32_t n) noexcept {
        if (n == 0u) return 0u;
        if (error || n > 64u || bitpos + n > total_bits()) {
            error = true;
            return 0u;
        }
        uint64_t val = 0u;
        uint32_t done = 0u;
        while (done < n) {
            const size_t byte_i = static_cast<size_t>(bitpos >> 3u);
            const uint32_t bit_i = static_cast<uint32_t>(bitpos & 7u);
            const uint32_t take = std::min<uint32_t>(n - done, 8u - bit_i);
            const uint64_t chunk = (static_cast<uint64_t>(data[byte_i]) >> bit_i) & ((1ull << take) - 1ull);
            val |= chunk << done;
            bitpos += take;
            done += take;
        }
        return val;
    }

    uint64_t read_vbr(uint32_t w) noexcept {
        if (error || w < 2u || w > 32u) {
            error = true;
            return 0u;
        }
        uint64_t res = 0u;
        uint32_t shift = 0u;
        const uint64_t mask = (1ull << (w - 1u)) - 1ull;
        const uint64_t high = 1ull << (w - 1u);
        while (!error) {
            const uint64_t piece = read(w);
            if (shift >= 64u) {
                error = true;
                return 0u;
            }
            res |= (piece & mask) << shift;
            if ((piece & high) == 0u) {
                return res;
            }
            shift += w - 1u;
        }
        return 0u;
    }

    void align32() noexcept {
        const uint64_t rem = bitpos & 31u;
        if (rem != 0u) {
            const uint64_t pad = 32u - rem;
            if (bitpos + pad > total_bits()) {
                error = true;
            } else {
                bitpos += pad;
            }
        }
    }
};

struct BitcodeWriter {
    std::vector<uint8_t> bytes;
    uint64_t bitpos{0u};

    void write_bits(uint64_t val, uint32_t n) {
        uint32_t done = 0u;
        while (done < n) {
            const size_t byte_i = static_cast<size_t>(bitpos >> 3u);
            const uint32_t bit_i = static_cast<uint32_t>(bitpos & 7u);
            if (byte_i >= bytes.size()) {
                bytes.resize(byte_i + 1u, 0u);
            }
            const uint32_t take = std::min<uint32_t>(n - done, 8u - bit_i);
            const uint8_t chunk = static_cast<uint8_t>((val >> done) & ((1ull << take) - 1ull));
            bytes[byte_i] |= static_cast<uint8_t>(chunk << bit_i);
            bitpos += take;
            done += take;
        }
    }

    void write_vbr(uint64_t val, uint32_t w) {
        const uint64_t mask = (1ull << (w - 1u)) - 1ull;
        const uint64_t high = 1ull << (w - 1u);
        while (true) {
            uint64_t piece = val & mask;
            val >>= (w - 1u);
            if (val != 0u) {
                write_bits(piece | high, w);
            } else {
                write_bits(piece, w);
                break;
            }
        }
    }

    void copy_bits(const uint8_t *src, uint64_t src_bit, uint64_t count) {
        if (count == 0u) return;
        if ((src_bit & 7u) == 0u && (bitpos & 7u) == 0u && (count >= 8u)) {
            const size_t full_bytes = static_cast<size_t>(count >> 3u);
            const size_t dst_byte = static_cast<size_t>(bitpos >> 3u);
            if (bytes.size() < dst_byte + full_bytes) {
                bytes.resize(dst_byte + full_bytes, 0u);
            }
            std::memcpy(bytes.data() + dst_byte, src + static_cast<size_t>(src_bit >> 3u), full_bytes);
            src_bit += static_cast<uint64_t>(full_bytes) * 8u;
            bitpos += static_cast<uint64_t>(full_bytes) * 8u;
            count -= static_cast<uint64_t>(full_bytes) * 8u;
        }
        while (count > 0u) {
            const size_t sb = static_cast<size_t>(src_bit >> 3u);
            const uint32_t sbi = static_cast<uint32_t>(src_bit & 7u);
            const uint32_t take = static_cast<uint32_t>(std::min<uint64_t>(count, 8u - sbi));
            const uint64_t chunk = (static_cast<uint64_t>(src[sb]) >> sbi) & ((1ull << take) - 1ull);
            write_bits(chunk, take);
            src_bit += take;
            count -= take;
        }
    }

    void align32() {
        const uint64_t rem = bitpos & 31u;
        if (rem != 0u) {
            write_bits(0u, static_cast<uint32_t>(32u - rem));
        }
    }

    void write_unabbrev(uint32_t abbrev_w, uint64_t rec_code, std::span<const uint64_t> ops) {
        write_bits(3u, abbrev_w);// UNABBREV_RECORD
        write_vbr(rec_code, 6u);
        write_vbr(ops.size(), 6u);
        for (uint64_t op : ops) {
            write_vbr(op, 6u);
        }
    }
};

struct BitcodeAbbrevOp {
    enum Kind : uint8_t { Literal, Fixed, VBR, Array, Char6, Blob };
    Kind kind{Literal};
    uint64_t data{0u};
};

// Patches DXIL LLVM 3.7 bitcode directly in O(|METADATA_BLOCK|) (~0.1 ms even
// on 350k-line shaders) without disassembling/re-parsing LLVM IR text.
// Copies all FUNCTION_BLOCKs verbatim by skipping subblocks via their 32-bit
// length words and appending new metadata nodes to !dx.entryPoints without
// shifting any existing value or metadata IDs.
[[nodiscard]] WaveSizeBitcodePatchResult inject_dxil_entry_wave_size_bitcode(
    std::span<const uint8_t> bc, uint32_t wave_size) {
    WaveSizeBitcodePatchResult res;
    if (bc.size() < 12u || (bc.size() & 3u) != 0u) {
        return res;
    }
    if (bc[0] != 0x42u || bc[1] != 0x43u || bc[2] != 0xC0u || bc[3] != 0xDEu) {
        return res;
    }
    constexpr const char kChar6[65] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._";

    BitcodeReader r{bc.data(), bc.size(), 32u, false};
    if (r.read(2u) != 1u) {// ENTER_SUBBLOCK
        return res;
    }
    const uint64_t top_bid = r.read_vbr(8u);
    const uint32_t mod_aw = static_cast<uint32_t>(r.read_vbr(4u));
    r.align32();
    if (r.error || top_bid != 8u || mod_aw < 2u || mod_aw > 32u) {
        return res;
    }
    const size_t mod_len_byte_pos = static_cast<size_t>(r.bitpos >> 3u);
    const uint32_t mod_words = static_cast<uint32_t>(r.read(32u));
    const size_t mod_end_byte = static_cast<size_t>(r.bitpos >> 3u) + static_cast<size_t>(mod_words) * 4u;
    if (r.error || mod_end_byte > bc.size()) {
        return res;
    }

    auto read_abbrev_def = [&](std::vector<BitcodeAbbrevOp> &ops) -> bool {
        const uint64_t nops = r.read_vbr(5u);
        if (r.error || nops > 4096u) return false;
        ops.clear();
        ops.reserve(static_cast<size_t>(nops));
        for (uint64_t i = 0u; i < nops; ++i) {
            if (r.read(1u) != 0u) {
                ops.push_back({BitcodeAbbrevOp::Literal, r.read_vbr(8u)});
            } else {
                const uint64_t enc = r.read(3u);
                if (enc == 1u) {
                    ops.push_back({BitcodeAbbrevOp::Fixed, r.read_vbr(5u)});
                } else if (enc == 2u) {
                    ops.push_back({BitcodeAbbrevOp::VBR, r.read_vbr(5u)});
                } else if (enc == 3u) {
                    ops.push_back({BitcodeAbbrevOp::Array, 0u});
                } else if (enc == 4u) {
                    ops.push_back({BitcodeAbbrevOp::Char6, 0u});
                } else if (enc == 5u) {
                    ops.push_back({BitcodeAbbrevOp::Blob, 0u});
                } else {
                    return false;
                }
            }
        }
        return !r.error;
    };

    auto read_abbrev_rec = [&](const std::vector<BitcodeAbbrevOp> &ops,
                               std::vector<uint64_t> &out,
                               bool &saw_blob) -> bool {
        out.clear();
        saw_blob = false;
        for (size_t i = 0u; i < ops.size(); ++i) {
            const auto &op = ops[i];
            switch (op.kind) {
                case BitcodeAbbrevOp::Literal:
                    out.push_back(op.data);
                    break;
                case BitcodeAbbrevOp::Fixed:
                    out.push_back(r.read(static_cast<uint32_t>(op.data)));
                    break;
                case BitcodeAbbrevOp::VBR:
                    out.push_back(r.read_vbr(static_cast<uint32_t>(op.data)));
                    break;
                case BitcodeAbbrevOp::Char6: {
                    const uint64_t idx = r.read(6u);
                    out.push_back(static_cast<uint64_t>(static_cast<unsigned char>(kChar6[idx & 63u])));
                    break;
                }
                case BitcodeAbbrevOp::Array: {
                    if (i + 1u >= ops.size()) return false;
                    const uint64_t n = r.read_vbr(6u);
                    if (r.error || n > bc.size()) return false;
                    const auto &elem = ops[++i];
                    out.reserve(out.size() + static_cast<size_t>(n));
                    for (uint64_t k = 0u; k < n; ++k) {
                        if (elem.kind == BitcodeAbbrevOp::Fixed) {
                            out.push_back(r.read(static_cast<uint32_t>(elem.data)));
                        } else if (elem.kind == BitcodeAbbrevOp::VBR) {
                            out.push_back(r.read_vbr(static_cast<uint32_t>(elem.data)));
                        } else if (elem.kind == BitcodeAbbrevOp::Char6) {
                            const uint64_t idx = r.read(6u);
                            out.push_back(static_cast<uint64_t>(static_cast<unsigned char>(kChar6[idx & 63u])));
                        } else if (elem.kind == BitcodeAbbrevOp::Literal) {
                            out.push_back(elem.data);
                        } else {
                            return false;
                        }
                    }
                    break;
                }
                case BitcodeAbbrevOp::Blob: {
                    saw_blob = true;
                    const uint64_t n = r.read_vbr(6u);
                    r.align32();
                    if (r.error || r.bitpos + n * 8u > r.total_bits()) {
                        r.error = true;
                        return false;
                    }
                    r.bitpos += n * 8u;
                    r.align32();
                    break;
                }
            }
            if (r.error) return false;
        }
        return !r.error;
    };

    std::unordered_map<uint64_t, std::vector<std::vector<BitcodeAbbrevOp>>> blockinfo;
    uint32_t i32_tid = UINT32_MAX;
    uint32_t type_cnt = 0u;
    uint64_t val_cnt = 0u;
    uint64_t cur_cst_tid = UINT64_MAX;
    std::unordered_map<int64_t, uint64_t> i32_val_to_id;
    std::unordered_map<uint64_t, int64_t> val_id_to_i32;

    uint64_t total_md_count = 0u;
    uint64_t existing_md_val_11 = UINT64_MAX;
    uint64_t existing_md_val_wave = UINT64_MAX;
    std::unordered_map<uint64_t, int64_t> md_id_to_i32;
    std::unordered_map<uint64_t, std::vector<uint64_t>> md_nodes;

    struct EpBlockInfo {
        bool found{false};
        size_t len_byte_pos{0u};
        size_t body_start_byte{0u};
        size_t body_end_byte{0u};
        uint32_t aw{0u};
        uint64_t ep_md_id{0u};
        uint64_t ep_name_start_bit{0u};
        uint64_t ep_named_end_bit{0u};
        uint64_t end_block_start_bit{0u};
        bool blob_after_ep{false};
        bool later_md_nodes{false};
    };
    EpBlockInfo ep_info;

    std::vector<uint64_t> rec_vals;
    std::vector<BitcodeAbbrevOp> tmp_abbrev;

    auto walk_subblock = [&](uint64_t bid, uint32_t aw,
                             size_t len_byte_pos, size_t body_start_byte, size_t body_end_byte) -> bool {
        std::vector<std::vector<BitcodeAbbrevOp>> local_abbrevs;
        uint64_t cur_bi_bid = UINT64_MAX;
        std::string pending_md_name;
        uint64_t pending_md_name_start_bit = 0u;
        bool pending_md_name_valid = false;
        bool in_ep_block = false;

        auto get_abbrev = [&](uint64_t abbrev_id) -> const std::vector<BitcodeAbbrevOp> * {
            if (abbrev_id < 4u) return nullptr;
            const size_t idx = static_cast<size_t>(abbrev_id - 4u);
            if (auto it = blockinfo.find(bid); it != blockinfo.end()) {
                if (idx < it->second.size()) {
                    return &it->second[idx];
                }
                const size_t lidx = idx - it->second.size();
                return lidx < local_abbrevs.size() ? &local_abbrevs[lidx] : nullptr;
            }
            return idx < local_abbrevs.size() ? &local_abbrevs[idx] : nullptr;
        };

        while (!r.error && r.bitpos < static_cast<uint64_t>(body_end_byte) * 8u) {
            const uint64_t rec_start_bit = r.bitpos;
            const uint64_t code = r.read(aw);
            if (r.error) return false;
            if (code == 0u) {// END_BLOCK
                if (in_ep_block) {
                    ep_info.end_block_start_bit = rec_start_bit;
                }
                r.align32();
                return !r.error && r.bitpos == static_cast<uint64_t>(body_end_byte) * 8u;
            }
            if (code == 1u) {// ENTER_SUBBLOCK (nested subblock inside subblock not expected here)
                const uint64_t sub_id = r.read_vbr(8u);
                (void)sub_id;
                (void)r.read_vbr(4u);
                r.align32();
                const uint32_t sub_words = static_cast<uint32_t>(r.read(32u));
                const uint64_t sub_end_bit = r.bitpos + static_cast<uint64_t>(sub_words) * 32u;
                if (r.error || sub_end_bit > static_cast<uint64_t>(body_end_byte) * 8u) return false;
                r.bitpos = sub_end_bit;
                pending_md_name_valid = false;
                continue;
            }
            if (code == 2u) {// DEFINE_ABBREV
                if (!read_abbrev_def(tmp_abbrev)) return false;
                if (bid == 0u) {
                    blockinfo[cur_bi_bid].push_back(tmp_abbrev);
                } else {
                    local_abbrevs.push_back(tmp_abbrev);
                }
                pending_md_name_valid = false;
                continue;
            }

            uint64_t rc = 0u;
            bool saw_blob = false;
            if (code == 3u) {// UNABBREV_RECORD
                rc = r.read_vbr(6u);
                const uint64_t nops = r.read_vbr(6u);
                if (r.error || nops > bc.size()) return false;
                rec_vals.resize(static_cast<size_t>(nops));
                for (uint64_t i = 0u; i < nops; ++i) {
                    rec_vals[static_cast<size_t>(i)] = r.read_vbr(6u);
                }
                if (r.error) return false;
            } else {
                const auto *ops = get_abbrev(code);
                if (ops == nullptr) return false;
                std::vector<uint64_t> raw;
                if (!read_abbrev_rec(*ops, raw, saw_blob) || raw.empty()) return false;
                rc = raw[0];
                rec_vals.assign(raw.begin() + 1, raw.end());
            }
            const uint64_t rec_end_bit = r.bitpos;
            if (in_ep_block && saw_blob) {
                ep_info.blob_after_ep = true;
            }

            if (bid == 0u) {
                if (rc == 1u && !rec_vals.empty()) {
                    cur_bi_bid = rec_vals[0];
                }
            } else if (bid == 17u) {// TYPE_BLOCK_ID_NEW
                if (rc != 1u && rc != 19u) {
                    if (rc == 7u && rec_vals.size() == 1u && rec_vals[0] == 32u) {
                        i32_tid = type_cnt;
                    }
                    ++type_cnt;
                }
            } else if (bid == 11u) {// CONSTANTS_BLOCK
                if (rc == 1u && !rec_vals.empty()) {
                    cur_cst_tid = rec_vals[0];
                } else {
                    if (i32_tid != UINT32_MAX && cur_cst_tid == i32_tid) {
                        if (rc == 2u) {
                            i32_val_to_id.try_emplace(0, val_cnt);
                            val_id_to_i32[val_cnt] = 0;
                        } else if (rc == 4u && !rec_vals.empty()) {
                            const uint64_t v = rec_vals[0];
                            if ((v & 1u) == 0u) {
                                const int64_t sv = static_cast<int64_t>(v >> 1u);
                                i32_val_to_id.try_emplace(sv, val_cnt);
                                val_id_to_i32[val_cnt] = sv;
                            }
                        }
                    }
                    ++val_cnt;
                }
            } else if (bid == 15u) {// METADATA_BLOCK
                if (rc == 4u) {// METADATA_NAME
                    pending_md_name.clear();
                    pending_md_name.reserve(rec_vals.size());
                    for (uint64_t ch : rec_vals) {
                        pending_md_name.push_back(static_cast<char>(ch & 0xFFu));
                    }
                    pending_md_name_start_bit = rec_start_bit;
                    pending_md_name_valid = true;
                } else if (rc == 10u) {// METADATA_NAMED_NODE
                    if (pending_md_name_valid && pending_md_name == "dx.entryPoints" &&
                        rec_vals.size() == 1u && !ep_info.found) {
                        ep_info.found = true;
                        ep_info.len_byte_pos = len_byte_pos;
                        ep_info.body_start_byte = body_start_byte;
                        ep_info.body_end_byte = body_end_byte;
                        ep_info.aw = aw;
                        ep_info.ep_md_id = rec_vals[0];
                        ep_info.ep_name_start_bit = pending_md_name_start_bit;
                        ep_info.ep_named_end_bit = rec_end_bit;
                        in_ep_block = true;
                    }
                    pending_md_name_valid = false;
                } else if (rc == 6u || rc == 11u) {// METADATA_KIND / METADATA_ATTACHMENT
                    pending_md_name_valid = false;
                } else {
                    pending_md_name_valid = false;
                    if (ep_info.found && !in_ep_block) {
                        ep_info.later_md_nodes = true;
                    }
                    const uint64_t md_id = total_md_count++;
                    if (rc == 2u && rec_vals.size() == 2u &&
                        i32_tid != UINT32_MAX && rec_vals[0] == i32_tid) {
                        if (auto it = val_id_to_i32.find(rec_vals[1]); it != val_id_to_i32.end()) {
                            md_id_to_i32[md_id] = it->second;
                            if (it->second == 11 && existing_md_val_11 == UINT64_MAX) {
                                existing_md_val_11 = md_id;
                            }
                            if (it->second == static_cast<int64_t>(wave_size) &&
                                existing_md_val_wave == UINT64_MAX) {
                                existing_md_val_wave = md_id;
                            }
                        }
                    } else if (rc == 3u || rc == 5u) {
                        md_nodes.emplace(md_id, rec_vals);
                    }
                }
            }
        }
        return false;
    };

    // Walk MODULE_BLOCK (id=8) top level, skipping FUNCTION_BLOCKs in O(1).
    std::vector<std::vector<BitcodeAbbrevOp>> mod_local_abbrevs;
    auto get_mod_abbrev = [&](uint64_t abbrev_id) -> const std::vector<BitcodeAbbrevOp> * {
        if (abbrev_id < 4u) return nullptr;
        const size_t idx = static_cast<size_t>(abbrev_id - 4u);
        if (auto it = blockinfo.find(8u); it != blockinfo.end()) {
            if (idx < it->second.size()) return &it->second[idx];
            const size_t lidx = idx - it->second.size();
            return lidx < mod_local_abbrevs.size() ? &mod_local_abbrevs[lidx] : nullptr;
        }
        return idx < mod_local_abbrevs.size() ? &mod_local_abbrevs[idx] : nullptr;
    };

    while (!r.error && r.bitpos < static_cast<uint64_t>(mod_end_byte) * 8u) {
        const uint64_t code = r.read(mod_aw);
        if (r.error) return res;
        if (code == 0u) {// END_BLOCK MODULE_BLOCK
            r.align32();
            break;
        }
        if (code == 1u) {// ENTER_SUBBLOCK
            const uint64_t sub_id = r.read_vbr(8u);
            const uint32_t sub_aw = static_cast<uint32_t>(r.read_vbr(4u));
            r.align32();
            const size_t sub_len_byte_pos = static_cast<size_t>(r.bitpos >> 3u);
            const uint32_t sub_words = static_cast<uint32_t>(r.read(32u));
            const size_t sub_start_byte = static_cast<size_t>(r.bitpos >> 3u);
            const size_t sub_end_byte = sub_start_byte + static_cast<size_t>(sub_words) * 4u;
            if (r.error || sub_aw < 2u || sub_aw > 32u || sub_end_byte > mod_end_byte) {
                return res;
            }
            if (sub_id == 0u || sub_id == 17u || sub_id == 11u || sub_id == 15u) {
                if (!walk_subblock(sub_id, sub_aw, sub_len_byte_pos, sub_start_byte, sub_end_byte)) {
                    return res;
                }
            } else {
                r.bitpos = static_cast<uint64_t>(sub_end_byte) * 8u;
            }
            continue;
        }
        if (code == 2u) {// DEFINE_ABBREV
            if (!read_abbrev_def(tmp_abbrev)) return res;
            mod_local_abbrevs.push_back(tmp_abbrev);
            continue;
        }
        uint64_t rc = 0u;
        bool saw_blob = false;
        if (code == 3u) {
            rc = r.read_vbr(6u);
            const uint64_t nops = r.read_vbr(6u);
            if (r.error || nops > bc.size()) return res;
            for (uint64_t i = 0u; i < nops; ++i) {
                (void)r.read_vbr(6u);
            }
            if (r.error) return res;
        } else {
            const auto *ops = get_mod_abbrev(code);
            if (ops == nullptr) return res;
            std::vector<uint64_t> raw;
            if (!read_abbrev_rec(*ops, raw, saw_blob) || raw.empty()) return res;
            rc = raw[0];
        }
        if (rc == 7u || rc == 8u || rc == 9u) {// GLOBALVAR / FUNCTION / ALIAS
            ++val_cnt;
        }
    }

    if (r.error || i32_tid == UINT32_MAX || !ep_info.found ||
        ep_info.end_block_start_bit == 0u || ep_info.blob_after_ep ||
        ep_info.later_md_nodes) {
        return res;
    }
    auto ep_it = md_nodes.find(ep_info.ep_md_id);
    if (ep_it == md_nodes.end() || ep_it->second.size() < 5u) {
        return res;
    }
    const auto &ep_ops = ep_it->second;
    std::vector<uint64_t> props_ops;
    if (ep_ops[4] != 0u) {
        auto props_it = md_nodes.find(ep_ops[4] - 1u);
        if (props_it == md_nodes.end()) {
            return res;
        }
        props_ops = props_it->second;
        for (size_t i = 0u; i + 1u < props_ops.size(); i += 2u) {
            if (props_ops[i] != 0u) {
                if (auto tag_it = md_id_to_i32.find(props_ops[i] - 1u); tag_it != md_id_to_i32.end()) {
                    if (tag_it->second == 11 || tag_it->second == 23) {
                        res.ok = true;
                        res.already_present = true;
                        return res;
                    }
                }
            }
        }
    }

    auto val11_it = i32_val_to_id.find(11);
    auto valw_it = i32_val_to_id.find(static_cast<int64_t>(wave_size));
    if (val11_it == i32_val_to_id.end() || valw_it == i32_val_to_id.end()) {
        // Constants not present in CONSTANTS_BLOCK (only on minimal shaders);
        // caller falls back to text IR assembly.
        return res;
    }

    BitcodeWriter bw;
    bw.bytes.reserve((ep_info.body_end_byte - ep_info.body_start_byte) + 128u);
    const uint64_t body_start_bit = static_cast<uint64_t>(ep_info.body_start_byte) * 8u;
    bw.copy_bits(bc.data(), body_start_bit, ep_info.ep_name_start_bit - body_start_bit);
    bw.copy_bits(bc.data(), ep_info.ep_named_end_bit,
                 ep_info.end_block_start_bit - ep_info.ep_named_end_bit);

    uint64_t next_md = total_md_count;
    uint64_t md_11 = existing_md_val_11;
    if (md_11 == UINT64_MAX) {
        md_11 = next_md++;
        const uint64_t ops_arr[2] = {i32_tid, val11_it->second};
        bw.write_unabbrev(ep_info.aw, 2u, ops_arr);
    }
    uint64_t md_wave = existing_md_val_wave;
    if (md_wave == UINT64_MAX) {
        md_wave = next_md++;
        const uint64_t ops_arr[2] = {i32_tid, valw_it->second};
        bw.write_unabbrev(ep_info.aw, 2u, ops_arr);
    }
    const uint64_t md_wave_tuple = next_md++;
    {
        const uint64_t ops_arr[1] = {md_wave + 1u};
        bw.write_unabbrev(ep_info.aw, 3u, ops_arr);
    }
    props_ops.push_back(md_11 + 1u);
    props_ops.push_back(md_wave_tuple + 1u);
    const uint64_t md_new_props = next_md++;
    bw.write_unabbrev(ep_info.aw, 3u, props_ops);

    std::vector<uint64_t> new_ep_ops = ep_ops;
    new_ep_ops[4] = md_new_props + 1u;
    const uint64_t md_new_ep = next_md++;
    bw.write_unabbrev(ep_info.aw, 3u, new_ep_ops);

    {
        constexpr const char kEpName[] = "dx.entryPoints";
        uint64_t name_ops[sizeof(kEpName) - 1u];
        for (size_t i = 0u; i < sizeof(kEpName) - 1u; ++i) {
            name_ops[i] = static_cast<uint64_t>(static_cast<unsigned char>(kEpName[i]));
        }
        bw.write_unabbrev(ep_info.aw, 4u, name_ops);
    }
    {
        const uint64_t named_ops[1] = {md_new_ep};
        bw.write_unabbrev(ep_info.aw, 10u, named_ops);
    }
    bw.write_bits(0u, ep_info.aw);// END_BLOCK
    bw.align32();
    if ((bw.bytes.size() & 3u) != 0u) {
        return res;
    }

    const uint32_t old_md_words = static_cast<uint32_t>((ep_info.body_end_byte - ep_info.body_start_byte) / 4u);
    const uint32_t new_md_words = static_cast<uint32_t>(bw.bytes.size() / 4u);
    const uint32_t new_mod_words = mod_words + new_md_words - old_md_words;

    res.patched_bc.resize(bc.size() - (ep_info.body_end_byte - ep_info.body_start_byte) + bw.bytes.size());
    std::memcpy(res.patched_bc.data(), bc.data(), ep_info.body_start_byte);
    std::memcpy(res.patched_bc.data() + ep_info.body_start_byte, bw.bytes.data(), bw.bytes.size());
    std::memcpy(res.patched_bc.data() + ep_info.body_start_byte + bw.bytes.size(),
                bc.data() + ep_info.body_end_byte,
                bc.size() - ep_info.body_end_byte);
    std::memcpy(res.patched_bc.data() + ep_info.len_byte_pos, &new_md_words, sizeof(uint32_t));
    std::memcpy(res.patched_bc.data() + mod_len_byte_pos, &new_mod_words, sizeof(uint32_t));
    res.ok = true;
    return res;
}

}// namespace

std::optional<CompileResult> ShaderCompiler::_try_compile_large_dxil_fast(
    vstd::string_view code,
    vstd::wstring_view profile,
    vstd::span<LPCWSTR> base_args,
    IDxcIncludeHandler *include_handler) const {
    if (compiler_module.dxc_create_instance == nullptr ||
        compiler_module.dxil_create_instance == nullptr ||
        compiler() == nullptr ||
        utils() == nullptr) {
        return std::nullopt;
    }
    // 1. Obtain cached rewritten -O3 pass list for this target profile.
    std::vector<std::wstring> passes;
    {
        std::lock_guard lock{_dxil_pass_cache_mutex};
        std::wstring key{profile.data(), profile.size()};
        if (auto it = _dxil_fast_pass_cache.find(key); it != _dxil_fast_pass_cache.end()) {
            passes = it->second;
        }
    }
    if (passes.empty()) {
        std::wstring key{profile.data(), profile.size()};
        constexpr std::string_view kTinyCompute = "[numthreads(32,1,1)] void main() {}\n";
        vstd::fixed_vector<LPCWSTR, 16> dump_args;
        vstd::push_back_all(
            dump_args,
            {L"-E", L"main", L"-T", key.c_str(),
             DXC_ARG_ALL_RESOURCES_BOUND,
             L"-enable-16bit-types",
             DXC_ARG_PACK_MATRIX_ROW_MAJOR,
             L"-HV 2021",
             L"-no-warnings",
             DXC_ARG_OPTIMIZATION_LEVEL3,
             L"-Odump"});
        auto dump_res = compile(kTinyCompute, dump_args);
        if (!dump_res.is_type_of<ComUniquePtr<IDxcBlob>>()) {
            return std::nullopt;
        }
        auto &dump_blob = dump_res.get<0>();
        std::string_view dump_text{
            reinterpret_cast<char const *>(dump_blob->GetBufferPointer()),
            dump_blob->GetBufferSize()};
        passes = parse_and_rewrite_dxil_o3_passes(dump_text);
        if (passes.empty()) {
            return std::nullopt;
        }
        std::lock_guard lock{_dxil_pass_cache_mutex};
        _dxil_fast_pass_cache.try_emplace(std::move(key), passes);
    }

    // 2. Frontend + High-Level DXIL codegen only (`-fcgl -fdisable-loc-tracking`).
    vstd::fixed_vector<LPCWSTR, 64> fcgl_args;
    for (auto arg : base_args) {
        fcgl_args.emplace_back(arg);
    }
    fcgl_args.emplace_back(L"-fdisable-loc-tracking");
    fcgl_args.emplace_back(L"-fcgl");
    auto fcgl_res = compile(code, fcgl_args, include_handler);
    if (!fcgl_res.is_type_of<ComUniquePtr<IDxcBlob>>()) {
        // Real HLSL syntax/semantic compile error — return it directly instead
        // of re-running slow -O3 on invalid HLSL.
        return fcgl_res;
    }
    auto hl_blob = std::move(fcgl_res.get<0>());

    // 3. Create thread-local COM optimizer, assembler, and dxil validator.
    IDxcOptimizer *opt_raw{nullptr};
    IDxcAssembler *asm_raw{nullptr};
    IDxcValidator *val_raw{nullptr};
    if (FAILED(compiler_module.dxc_create_instance(
            CLSID_DxcOptimizer, IID_PPV_ARGS(&opt_raw))) ||
        opt_raw == nullptr) {
        return std::nullopt;
    }
    ComUniquePtr<IDxcOptimizer> optimizer{opt_raw};
    if (FAILED(compiler_module.dxc_create_instance(
            CLSID_DxcAssembler, IID_PPV_ARGS(&asm_raw))) ||
        asm_raw == nullptr) {
        return std::nullopt;
    }
    ComUniquePtr<IDxcAssembler> assembler{asm_raw};
    if (FAILED(compiler_module.dxil_create_instance(
            CLSID_DxcValidator, IID_PPV_ARGS(&val_raw))) ||
        val_raw == nullptr) {
        return std::nullopt;
    }
    ComUniquePtr<IDxcValidator> validator{val_raw};

    // 4. Run rewritten -O3 pass pipeline via IDxcOptimizer.
    std::vector<LPCWSTR> pass_ptrs;
    pass_ptrs.reserve(passes.size());
    for (auto &p : passes) {
        pass_ptrs.emplace_back(p.c_str());
    }
    IDxcBlob *opt_blob_raw{nullptr};
    IDxcBlobEncoding *opt_txt_raw{nullptr};
    HRESULT hr = optimizer->RunOptimizer(
        hl_blob.get(),
        pass_ptrs.data(),
        static_cast<UINT32>(pass_ptrs.size()),
        &opt_blob_raw,
        &opt_txt_raw);
    ComUniquePtr<IDxcBlobEncoding> opt_txt{opt_txt_raw};
    ComUniquePtr<IDxcBlob> opt_blob{opt_blob_raw};
    if (FAILED(hr) || !opt_blob) {
        return std::nullopt;
    }

    // 4b. Preserve [WaveSize(N)] lost by DXC's -fcgl (hlsl-hlemit drops
    // WaveSize from !dx.fnprops, causing variable-wave drivers like AMD RDNA
    // to fall back to Wave64 on large shaders).
    // Patch the LLVM 3.7 bitcode METADATA_BLOCK directly in O(|METADATA_BLOCK|)
    // (~0.1 ms even on 350k-line shaders), falling back to IR disassembly only
    // when the tiny shader lacks i32 11 / i32 N in its CONSTANTS_BLOCK.
    IDxcBlob *asm_input_blob = opt_blob.get();
    ComUniquePtr<IDxcBlobEncoding> patched_ir_blob;
    if (const uint32_t wave_size = parse_entry_wave_size({code.data(), code.size()});
        wave_size != 0u) {
        bool preserved = false;
        std::span<const uint8_t> opt_bytes{
            static_cast<const uint8_t *>(opt_blob->GetBufferPointer()),
            opt_blob->GetBufferSize()};
        auto bc_patch = inject_dxil_entry_wave_size_bitcode(opt_bytes, wave_size);
        if (bc_patch.ok) {
            if (bc_patch.already_present) {
                preserved = true;
            } else {
                IDxcBlobEncoding *pblob_raw{nullptr};
                if (SUCCEEDED(utils()->CreateBlob(
                        bc_patch.patched_bc.data(),
                        static_cast<UINT32>(bc_patch.patched_bc.size()),
                        0u,
                        &pblob_raw)) &&
                    pblob_raw != nullptr) {
                    patched_ir_blob = ComUniquePtr<IDxcBlobEncoding>{pblob_raw};
                    asm_input_blob = patched_ir_blob.get();
                    preserved = true;
                }
            }
        }
        if (!preserved) {
            DxcBuffer opt_buf{opt_blob->GetBufferPointer(), opt_blob->GetBufferSize(), 0};
            IDxcResult *dis_res_raw{nullptr};
            if (SUCCEEDED(compiler()->Disassemble(&opt_buf, IID_PPV_ARGS(&dis_res_raw))) &&
                dis_res_raw != nullptr) {
                ComUniquePtr<IDxcResult> dis_res{dis_res_raw};
                IDxcBlobUtf8 *dis_txt_raw{nullptr};
                if (SUCCEEDED(dis_res->GetOutput(DXC_OUT_DISASSEMBLY, IID_PPV_ARGS(&dis_txt_raw), nullptr)) &&
                    dis_txt_raw != nullptr) {
                    ComUniquePtr<IDxcBlobUtf8> dis_txt{dis_txt_raw};
                    std::string_view ir_view{
                        static_cast<char const *>(dis_txt->GetBufferPointer()),
                        dis_txt->GetBufferSize()};
                    auto patch = inject_dxil_entry_wave_size(ir_view, wave_size);
                    if (patch.ok) {
                        if (patch.already_present) {
                            preserved = true;
                        } else {
                            IDxcBlobEncoding *pblob_raw{nullptr};
                            if (SUCCEEDED(utils()->CreateBlob(
                                    patch.patched_ir.data(),
                                    static_cast<UINT32>(patch.patched_ir.size()),
                                    DXC_CP_ACP,
                                    &pblob_raw)) &&
                                pblob_raw != nullptr) {
                                patched_ir_blob = ComUniquePtr<IDxcBlobEncoding>{pblob_raw};
                                asm_input_blob = patched_ir_blob.get();
                                preserved = true;
                            }
                        }
                    }
                }
            }
        }
        if (!preserved) {
            return std::nullopt;
        }
    }

    // 5. Assemble DXIL bitcode/IR into a DXBC container.
    IDxcOperationResult *asm_res_raw{nullptr};
    hr = assembler->AssembleToContainer(asm_input_blob, &asm_res_raw);
    ComUniquePtr<IDxcOperationResult> asm_res{asm_res_raw};
    if (FAILED(hr) || !asm_res) {
        return std::nullopt;
    }
    HRESULT asm_status = E_FAIL;
    if (FAILED(asm_res->GetStatus(&asm_status)) || FAILED(asm_status)) {
        return std::nullopt;
    }
    IDxcBlob *container_raw{nullptr};
    if (FAILED(asm_res->GetResult(&container_raw)) || container_raw == nullptr) {
        return std::nullopt;
    }
    ComUniquePtr<IDxcBlob> container_blob{container_raw};

    // 6. Validate and sign DXIL container in-place via dxil.dll.
    IDxcOperationResult *val_res_raw{nullptr};
    hr = validator->Validate(
        container_blob.get(),
        DxcValidatorFlags_InPlaceEdit,
        &val_res_raw);
    ComUniquePtr<IDxcOperationResult> val_res{val_res_raw};
    if (FAILED(hr) || !val_res) {
        return std::nullopt;
    }
    HRESULT val_status = E_FAIL;
    if (FAILED(val_res->GetStatus(&val_status)) || FAILED(val_status)) {
        return std::nullopt;
    }
    return CompileResult{std::move(container_blob)};
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
    if (!spirv && optimize && !debug && profile.starts_with(L"cs_") &&
        code.size() >= large_dxil_fast_opt_threshold()) {
        if (include_dirs.empty()) {
            if (auto fast = _try_compile_large_dxil_fast(code, profile, args, nullptr)) {
                return std::move(*fast);
            }
        } else {
            FileSystemIncludeHandler handler{utils(), include_dirs};
            if (auto fast = _try_compile_large_dxil_fast(code, profile, args, &handler)) {
                return std::move(*fast);
            }
        }
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
