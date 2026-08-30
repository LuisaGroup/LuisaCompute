// lc_compile_builtin: AOT-compile a raw HLSL builtin source into the bytecode
// containers read by the backend builtin-kernel loaders.
//
// `src/backends/dx/Shader/BuiltinKernel.cpp` and `src/backends/vk/builtin_kernel.cpp`
// pick their helper shaders up either as embedded blobs (the
// `src/backends/common/hlsl/builtin/*.dxil` files) or through the shader
// serializers. This tool regenerates those containers from the canonical HLSL
// sources with the exact same compiler the runtime uses:
//
//   * `dx` writes the DX `ShaderSerializer` v5 artifact
//     (header + serialized root signature + DXBC + properties), which
//     `ComputeShader::compile_compute(..., CacheType::Internal, ...)` loads.
//   * `vk` writes the VK `ShaderSerializer` v10 compute artifact
//     (header + properties + SPIR-V), i.e. `SerdeType::kBuiltin` bytecode.
//
// The resource properties - and with them the root-signature/descriptor layout -
// are parsed from the `:register(...)` annotations of the input source, and the
// block size from its `[numthreads(...)]` attribute, so the emitted interface
// matches the hand-written tables in `BuiltinKernel`.
//
// Usage:
//   lc-compile-builtin <dx|vk> <input-hlsl> <output> [options]
// e.g.:
//   lc-compile-builtin dx src/backends/common/hlsl/builtin/bindless_upload.bytes \
//                       src/backends/common/hlsl/builtin/load_bdls.dxil
//   lc-compile-builtin vk src/backends/common/hlsl/builtin/bindless_upload.bytes \
//                       src/backends/common/hlsl/builtin/load_bdls_vk.dxil
//
// Options:
//   --entry <name>         dxc entry point (default: main)
//   --shader-model <n>     packed shader model, e.g. 62 for cs_6_2 (default: 62)
//   --block-size <x[,y,z]> override the parsed [numthreads(...)] block size
//   --no-optimize          compile with dxc -Od instead of -O3
//   --raw                  write the bare dxc blob instead of a serializer artifact
//   --name <label>         shader label used in log messages
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <luisa/core/basic_traits.h>
#ifdef _WIN32
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <Windows.h>
#   include <d3d12.h>
#endif
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/context.h>
#include <luisa/vstl/md5.h>
#include "../common/hlsl/shader_compiler.h"
#include "../common/hlsl/shader_property.h"
using namespace luisa;
using namespace luisa::compute;
using namespace lc;// hlsl::ShaderCompiler, hlsl::Property, ...
using namespace std::literals;
namespace {
constexpr auto kDefaultKernelName = "test_builtin";
/**
 * \brief A single `:register(...)` annotation found in a builtin HLSL source.
 */
struct Binding {
    hlsl::ShaderVariableType type;
    char letter;
    uint32_t space;
    uint32_t register_index;
    uint32_t array_size;
};
/**
 * \brief Compile request parsed from the command line.
 */
struct Options {
    bool is_spirv{false};
    std::filesystem::path input;
    std::filesystem::path output;
    std::string name{kDefaultKernelName};
    std::string entry{"main"};
    uint32_t shader_model{62u};
    uint32_t block_size[3]{1u, 1u, 1u};
    bool has_block_size{false};
    bool optimize{true};
    bool raw{false};
};
[[nodiscard]] std::filesystem::path absolute_path(std::string_view p) {
    auto path = std::filesystem::path{p};
    return path.is_absolute() ? path : std::filesystem::absolute(path);
}
void print_usage(const char *exe) {
    LUISA_INFO(
        "Usage: {} <dx|vk> <input-hlsl> <output> [options]\n"
        "  dx|vk              - target bytecode dialect\n"
        "  input-hlsl         - raw HLSL builtin source, e.g. src/backends/common/hlsl/builtin/bindless_upload.bytes\n"
        "  output             - destination artifact path (e.g. load_bdls.dxil or load_bdls_vk.dxil)\n"
        "  --entry <name>         dxc entry point (default: main)\n"
        "  --shader-model <n>     packed shader model, 62 == cs_6_2 (default: 62)\n"
        "  --block-size <x[,y,z]> override [numthreads(...)]\n"
        "  --no-optimize          compile with -Od instead of -O3\n"
        "  --raw                  write the bare dxc blob instead of a serializer artifact\n"
        "  --name <label>         shader label used in log messages",
        exe);
}
/**
 * \brief Parse the command line. Returns false (without throwing) on usage errors.
 */
[[nodiscard]] bool parse_options(int argc, char *argv[], Options &opt) noexcept {
    // Usage problems are reported without aborting so the caller gets exit code 1.
    auto usage_error = [&](std::string_view message) noexcept {
        LUISA_WARNING("{}", message);
    };
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        auto arg = std::string_view{argv[i]};
        if (arg == "--help" || arg == "-h") {
            print_usage(argc > 0 ? argv[0] : "lc-compile-builtin");
            return false;
        }
        if (arg == "--no-optimize") {
            opt.optimize = false;
            continue;
        }
        if (arg == "--raw") {
            opt.raw = true;
            continue;
        }
        if (arg.front() != '-') {
            positional.emplace_back(arg);
            continue;
        }
        // Value-carrying options.
        if (i + 1 >= argc) {
            usage_error("option "s + std::string{arg} + " requires a value.");
            return false;
        }
        auto value = std::string_view{argv[++i]};
        if (arg == "--entry") {
            opt.entry = std::string{value};
        } else if (arg == "--name") {
            opt.name = std::string{value};
        } else if (arg == "--shader-model") {
            opt.shader_model = static_cast<uint32_t>(std::strtoul(value.data(), nullptr, 10));
        } else if (arg == "--block-size") {
            uint32_t dims[3]{1u, 1u, 1u};
            size_t index{};
            size_t pos{};
            while (index < 3u) {
                auto comma = value.find(',', pos);
                auto token = value.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
                dims[index++] = static_cast<uint32_t>(std::strtoul(token.data(), nullptr, 10));
                if (comma == std::string_view::npos) { break; }
                pos = comma + 1u;
            }
            std::memcpy(opt.block_size, dims, sizeof(dims));
            opt.has_block_size = true;
        } else {
            usage_error("unknown option: " + std::string{arg});
            return false;
        }
    }
    if (positional.size() != 3u) {
        print_usage(argc > 0 ? argv[0] : "lc-compile-builtin");
        return false;
    }
    auto backend = std::string_view{positional[0]};
    if (backend == "dx") {
#ifndef _WIN32
        usage_error("the DX builtin artifact requires a Windows host (d3d12 root-signature serialization).");
        return false;
#else
        opt.is_spirv = false;
#endif
    } else if (backend == "vk") {
        opt.is_spirv = true;
    } else {
        usage_error("unknown backend '" + std::string{backend} + "': expected 'dx' or 'vk'.");
        return false;
    }
    if (opt.input = absolute_path(positional[1]); !std::filesystem::is_regular_file(opt.input)) {
        usage_error("input HLSL file not found: " + opt.input.string());
        return false;
    }
    opt.output = absolute_path(positional[2]);
    return true;
}
[[nodiscard]] std::string read_file(const std::filesystem::path &path) {
    std::ifstream f{path, std::ios::binary};
    if (!f) {
        LUISA_ERROR("Failed to open input HLSL file: {}", path.string());
    }
    return std::string{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
}
[[nodiscard]] bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}
[[nodiscard]] bool is_digit(char c) noexcept {
    return c >= '0' && c <= '9';
}
[[nodiscard]] bool is_ident_char(char c) noexcept {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}
/**
 * \brief Parse `[numthreads(x, y, z)]`. Returns false when the sizes are not
 * literals, e.g. the `THREAD_GROUP_SIZE` macro used by the BC encoders.
 */
[[nodiscard]] bool parse_block_size(std::string_view src, uint32_t block[3]) noexcept {
    auto pos = src.find("numthreads(");
    if (pos == std::string_view::npos) { return false; }
    pos += 11u;
    for (uint32_t i = 0u; i < 3u; ++i) {
        while (pos < src.size() && is_space(src[pos])) { ++pos; }
        if (pos >= src.size() || !is_digit(src[pos])) { return false; }
        uint32_t value{};
        while (pos < src.size() && is_digit(src[pos])) {
            value = value * 10u + static_cast<uint32_t>(src[pos++] - '0');
        }
        block[i] = value;
        if (i < 2u) {
            while (pos < src.size() && is_space(src[pos])) { ++pos; }
            if (pos >= src.size() || src[pos] != ',') { return false; }
            ++pos;
        }
    }
    return true;
}
// Resource type keywords, ordered so that the most specific match comes first:
// `RWStructuredBuffer` must never resolve to the `StructuredBuffer` it contains.
constexpr std::pair<std::string_view, hlsl::ShaderVariableType> kResourceKeywords[] = {
    {"RWStructuredBuffer", hlsl::ShaderVariableType::RWStructuredBuffer},
    {"ConstantBuffer", hlsl::ShaderVariableType::CBVBufferHeap},
    {"StructuredBuffer", hlsl::ShaderVariableType::StructuredBuffer},
    {"RWTexture3D", hlsl::ShaderVariableType::UAVTextureHeap},
    {"RWTexture2D", hlsl::ShaderVariableType::UAVTextureHeap},
    {"SamplerState", hlsl::ShaderVariableType::SamplerHeap},
    {"Texture3D", hlsl::ShaderVariableType::SRVTextureHeap},
    {"Texture2D", hlsl::ShaderVariableType::SRVTextureHeap},
    {"cbuffer", hlsl::ShaderVariableType::ConstantBuffer},
};
/**
 * \brief Order in which bindings become root parameters / descriptor slots:
 * constant buffers, SRV, UAV, then samplers - matching the hand-written
 * property tables in `BuiltinKernel`.
 */
[[nodiscard]] uint32_t binding_rank(const Binding &b) noexcept {
    switch (b.letter) {
        case 'b':
            return b.type == hlsl::ShaderVariableType::ConstantValue ? 1u : 0u;
        case 't':
            return 2u;
        case 'u':
            return 3u;
        case 's':
            return 4u;
        default:
            return 5u;
    }
}
/**
 * \brief Collect every `:register(<kind><index>[, space<n>])` annotation.
 */
[[nodiscard]] std::vector<Binding> parse_bindings(std::string_view src) {
    std::vector<Binding> bindings;
    size_t search_from{};
    while (search_from < src.size()) {
        auto rp = src.find("register", search_from);
        if (rp == std::string_view::npos) { break; }
        search_from = rp + 8u;
        // Skip a leading `:` and whitespace, then the opening parenthesis.
        auto p = rp + 8u;
        while (p < src.size() && (is_space(src[p]) || src[p] == ':')) { ++p; }
        if (p >= src.size() || src[p] != '(') { continue; }
        ++p;
        if (p >= src.size()) { continue; }
        auto letter = src[p];
        if (letter != 'b' && letter != 't' && letter != 'u' && letter != 's') { continue; }
        ++p;
        auto reg_begin = p;
        while (p < src.size() && is_digit(src[p])) { ++p; }
        if (p == reg_begin) { continue; }
        uint32_t register_index{};
        for (auto q = reg_begin; q < p; ++q) {
            register_index = register_index * 10u + static_cast<uint32_t>(src[q] - '0');
        }
        uint32_t space{};
        uint32_t array_size{1u};
        {
            // Optional `, space<n>` suffix.
            auto save = p;
            while (p < src.size() && is_space(src[p])) { ++p; }
            if (p < src.size() && src[p] == ',') {
                ++p;
                while (p < src.size() && is_space(src[p])) { ++p; }
                if (src.substr(p, 5u) == "space") {
                    p += 5u;
                    auto digits = p;
                    while (p < src.size() && is_digit(src[p])) { ++p; }
                    for (auto q = digits; q < p; ++q) {
                        space = space * 10u + static_cast<uint32_t>(src[q] - '0');
                    }
                } else {
                    p = save;
                }
            } else {
                p = save;
            }
        }
        // The declaration is the text between the previous statement terminator
        // and the annotation.
        auto decl_start = src.rfind(';', rp);
        if (auto brace = src.rfind('}', rp); brace != std::string_view::npos && brace > decl_start) {
            decl_start = brace;
        }
        decl_start = decl_start == std::string_view::npos ? 0u : decl_start + 1u;
        auto decl = src.substr(decl_start, rp - decl_start);
        std::string_view keyword;
        hlsl::ShaderVariableType type{};
        size_t after_keyword{};
        for (size_t i = 0u; i < decl.size() && keyword.empty(); ++i) {
            if (i > 0u && is_ident_char(decl[i - 1u])) { continue; }
            for (auto [candidate, candidate_type] : kResourceKeywords) {
                if (decl.substr(i, candidate.size()) != candidate) { continue; }
                if (auto q = i + candidate.size(); q < decl.size() && is_ident_char(decl[q])) { continue; }
                keyword = candidate;
                type = candidate_type;
                after_keyword = i + candidate.size();
                break;
            }
        }
        if (keyword.empty()) { continue; }
        // A `[` after the keyword (and after the `<...>` template arguments) is
        // the variable's array size.
        auto lt = decl.find('<', after_keyword);
        auto search_from_bracket = lt == std::string_view::npos ? after_keyword : decl.find('>', lt);
        if (search_from_bracket != std::string_view::npos && search_from_bracket < decl.size()) {
            if (auto lb = decl.find('[', search_from_bracket); lb != std::string_view::npos) {
                auto q = lb + 1u;
                while (q < decl.size() && is_space(decl[q])) { ++q; }
                uint32_t count{};
                bool any = false;
                while (q < decl.size() && is_digit(decl[q])) {
                    count = count * 10u + static_cast<uint32_t>(decl[q++] - '0');
                    any = true;
                }
                if (any && q < decl.size() && decl[q] == ']') { array_size = count; }
            }
        }
        bindings.emplace_back(Binding{type, letter, space, register_index, array_size});
    }
    std::stable_sort(bindings.begin(), bindings.end(), [](const Binding &a, const Binding &b) noexcept {
        if (auto ra = binding_rank(a), rb = binding_rank(b); ra != rb) { return ra < rb; }
        if (a.space != b.space) { return a.space < b.space; }
        return a.register_index < b.register_index;
    });
    return bindings;
}
[[nodiscard]] const uint8_t *empty_bytes() noexcept {
    static constexpr uint8_t value{};
    return &value;
}
[[nodiscard]] vstd::MD5 md5_of(std::string_view s) {
    return vstd::MD5{vstd::span<uint8_t const>{reinterpret_cast<uint8_t const *>(s.data()), s.size()}};
}
[[nodiscard]] vstd::MD5 md5_of(const void *data, size_t size) {
    return vstd::MD5{vstd::span<uint8_t const>{reinterpret_cast<uint8_t const *>(data), size}};
}
[[nodiscard]] vstd::MD5 md5_empty() { return md5_of(empty_bytes(), 0u); }
/**
 * \brief Translate a packed shader model (e.g. 62) into the dxc profile string
 * (e.g. L"cs_6_2"), exactly like `hlsl::GetSM` does.
 */
[[nodiscard]] std::wstring shader_model_profile(uint32_t shader_model) {
    auto major = std::to_wstring(shader_model / 10u);
    auto minor = std::to_wstring(shader_model % 10u);
    return L"cs_" + major + L"_" + minor;
}
/**
 * \brief Build the dxc argument list the HLSL backends use, so the emitted
 * bytecode is identical to a runtime-JIT builtin (see `AddCompileFlags` and
 * `ShaderCompiler::compile_compute` in `src/backends/common/hlsl/shader_compiler.cpp`).
 */
[[nodiscard]] std::vector<std::wstring> dxc_arguments(const Options &opt) {
    auto profile = shader_model_profile(opt.shader_model);
    std::vector<std::wstring> args;
    if (opt.is_spirv) {
        args.emplace_back(L"-spirv");
        args.emplace_back(L"/DSPV");
        if (opt.shader_model > 65u) {
            args.emplace_back(L"-fspv-target-env=vulkan1.3");
        } else if (opt.shader_model > 60u) {
            args.emplace_back(L"-fspv-target-env=vulkan1.2");
        }
    }
    args.emplace_back(L"-T");
    args.push_back(profile);
    args.emplace_back(L"-E");
    args.emplace_back(opt.entry.empty() ? std::wstring{L"main"} : std::wstring{opt.entry.begin(), opt.entry.end()});
    // AddCompileFlags(): all resources bound, 16-bit types, row-major matrices,
    // no forced flow control, shader model 2021 HLSL, and no warnings in release.
    args.emplace_back(L"-all_resources_bound");
    args.emplace_back(L"-enable-16bit-types");
    args.emplace_back(L"-Zpr");
    args.emplace_back(L"-Gfa");
    args.emplace_back(L"-HV 2021");
    args.emplace_back(opt.optimize ? L"-O3" : L"-Od");
    if (opt.optimize) {
        args.emplace_back(L"-no-warnings");
    }
    return args;
}
#ifdef _WIN32
/**
 * \brief Serialize the root signature exactly like `dx::ShaderSerializer` does
 * (root signature 1.0, one descriptor range per heap-style property).
 */
[[nodiscard]] std::vector<std::byte> serialize_root_signature(std::span<const hlsl::Property> properties) {
    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
    ranges.reserve(properties.size());
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    parameters.reserve(properties.size());
    for (auto &&property : properties) {
        switch (property.type) {
            case hlsl::ShaderVariableType::SRVTextureHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
            case hlsl::ShaderVariableType::CBVBufferHeap:
            case hlsl::ShaderVariableType::SamplerHeap:
            case hlsl::ShaderVariableType::UAVTextureHeap:
            case hlsl::ShaderVariableType::UAVBufferHeap: {
                auto range_type = [&]() noexcept {
                    switch (property.type) {
                        case hlsl::ShaderVariableType::SRVTextureHeap:
                        case hlsl::ShaderVariableType::SRVBufferHeap: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                        case hlsl::ShaderVariableType::CBVBufferHeap: return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                        case hlsl::ShaderVariableType::UAVTextureHeap:
                        case hlsl::ShaderVariableType::UAVBufferHeap: return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                        default: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
                    }
                }();
                auto range = D3D12_DESCRIPTOR_RANGE{};
                range.RangeType = range_type;
                range.NumDescriptors = property.array_size;
                range.BaseShaderRegister = property.register_index;
                range.RegisterSpace = property.space_index;
                range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                ranges.emplace_back(range);
                auto parameter = D3D12_ROOT_PARAMETER{};
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameter.DescriptorTable.NumDescriptorRanges = 1u;
                parameter.DescriptorTable.pDescriptorRanges = &ranges.back();
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                parameters.emplace_back(parameter);
            } break;
            case hlsl::ShaderVariableType::ConstantBuffer: {
                auto parameter = D3D12_ROOT_PARAMETER{};
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                parameter.Descriptor.ShaderRegister = property.register_index;
                parameter.Descriptor.RegisterSpace = property.space_index;
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                parameters.emplace_back(parameter);
            } break;
            case hlsl::ShaderVariableType::ConstantValue: {
                // `ShaderSerializer` forwards space_index as the 32-bit value count.
                auto parameter = D3D12_ROOT_PARAMETER{};
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
                parameter.Constants.Num32BitValues = property.space_index;
                parameter.Constants.ShaderRegister = property.register_index;
                parameter.Constants.RegisterSpace = 0u;
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                parameters.emplace_back(parameter);
            } break;
            case hlsl::ShaderVariableType::StructuredBuffer: {
                auto parameter = D3D12_ROOT_PARAMETER{};
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                parameter.Descriptor.ShaderRegister = property.register_index;
                parameter.Descriptor.RegisterSpace = property.space_index;
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                parameters.emplace_back(parameter);
            } break;
            case hlsl::ShaderVariableType::RWStructuredBuffer: {
                auto parameter = D3D12_ROOT_PARAMETER{};
                parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
                parameter.Descriptor.ShaderRegister = property.register_index;
                parameter.Descriptor.RegisterSpace = property.space_index;
                parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                parameters.emplace_back(parameter);
            } break;
            default:
                LUISA_ERROR("Unsupported builtin property type {} in the DX root signature.",
                            static_cast<int>(property.type));
        }
    }
    auto desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0.NumParameters = static_cast<UINT>(parameters.size());
    desc.Desc_1_0.pParameters = parameters.empty() ? nullptr : parameters.data();
    desc.Desc_1_0.NumStaticSamplers = 0u;
    desc.Desc_1_0.pStaticSamplers = nullptr;
    desc.Desc_1_0.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ID3DBlob *blob{nullptr};
    ID3DBlob *error{nullptr};
    if (auto hr = D3D12SerializeVersionedRootSignature(&desc, &blob, &error); FAILED(hr)) {
        auto message = error != nullptr && error->GetBufferSize() > 0u
                           ? std::string_view{reinterpret_cast<char const *>(error->GetBufferPointer()), error->GetBufferSize()}
                           : std::string_view{"unknown error"};
        if (error != nullptr) { error->Release(); }
        LUISA_ERROR("Failed to serialize the builtin root signature (hr=0x{:x}): {}", static_cast<uint32_t>(hr), message);
    }
    if (error != nullptr) { error->Release(); }
    auto result = std::vector<std::byte>{
        reinterpret_cast<std::byte const *>(blob->GetBufferPointer()),
        reinterpret_cast<std::byte const *>(blob->GetBufferPointer()) + blob->GetBufferSize()};
    blob->Release();
    return result;
}
#endif
/**
 * \brief The `dx::shader_ser::Header` layout written by `ShaderSerializer::Serialize`.
 */
struct DxShaderSerHeader {
    uint64_t header_version;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    uint64_t root_sig_bytes;
    uint64_t code_bytes;
    uint32_t block_size[3];
    uint32_t property_count;
    uint32_t bindless_count;
    uint32_t kernel_arg_count;
    uint32_t printer_count;
    uint32_t validation_count;
};
static_assert(sizeof(DxShaderSerHeader) == 88u);
/**
 * \brief The `vk::detail::ShaderSerHeader` (v10) layout written by
 * `vk::ShaderSerializer::serialize_bytecode`.
 */
struct VkShaderSerHeader {
    uint64_t header_ver;
    uint32_t pipeline_ver;
    vstd::MD5 md5;
    vstd::MD5 type_md5;
    vstd::MD5 property_md5;
    vstd::MD5 argument_md5;
    vstd::MD5 spv_md5;
    uint64_t property_size;
    uint64_t spv_byte_size;
    uint32_t block_size[3];
    uint32_t kernel_arg_count;
    uint32_t printer_count;
    uint32_t printer_size_bytes;
    vstd::MD5 printer_md5;
    uint32_t validation_count;
    uint32_t required_subgroup_size;
    uint64_t constant_ubo_size;
    vstd::MD5 constant_ubo_md5;
    uint8_t use_bindless_buffer;
    uint8_t use_bindless_tex2d;
    uint8_t use_bindless_tex3d;
    uint8_t codegen_dialect;
    uint64_t required_spirv_features;
    vstd::MD5 semantic_header_md5;
};
static_assert(alignof(VkShaderSerHeader) == 8u);
static_assert(sizeof(VkShaderSerHeader) == 216u);
constexpr uint64_t kDxHeaderVersion = 5ull;
constexpr uint32_t kVkShaderSerVersion = 10u;
constexpr uint32_t kVkXirPipelineVersion = 4u;
constexpr uint64_t kVkComputeHeaderDomainTag = 0x314844524448534Cull;// "LSHDRH1"
constexpr uint8_t kVkDialectHlslSpirv = 0u;
/**
 * \brief Little-endian byte sink used for the VK canonical semantic-header digest.
 */
class CanonicalBytes {
public:
    void u8(uint8_t v) noexcept { bytes.push_back(v); }
    void u32(uint32_t v) noexcept {
        for (auto s = 0u; s < 32u; s += 8u) { push(static_cast<uint8_t>(v >> s)); }
    }
    void u64(uint64_t v) noexcept {
        for (auto s = 0u; s < 64u; s += 8u) { push(static_cast<uint8_t>(v >> s)); }
    }
    void md5(const vstd::MD5 &v) noexcept {
        auto data = v.to_binary();
        u64(data.data0);
        u64(data.data1);
    }
    [[nodiscard]] vstd::MD5 digest() const { return md5_of(bytes.data(), bytes.size()); }
    [[nodiscard]] size_t size() const noexcept { return bytes.size(); }
private:
    void push(uint8_t v) noexcept { bytes.push_back(v); }
    std::vector<uint8_t> bytes;
};
/**
 * \brief Hash the property table as four 32-bit words per property (type
 * widened), so that the digest never depends on host struct padding.
 */
[[nodiscard]] vstd::MD5 vk_property_md5(std::span<const hlsl::Property> properties) {
    std::vector<uint8_t> records;
    records.reserve(properties.size() * 16u);
    auto append = [&](uint32_t v) noexcept {
        for (auto s = 0u; s < 32u; s += 8u) { records.push_back(static_cast<uint8_t>(v >> s)); }
    };
    for (auto &&property : properties) {
        append(static_cast<uint32_t>(property.type));
        append(property.space_index);
        append(property.register_index);
        append(property.array_size);
    }
    return md5_of(records.data(), records.size());
}
[[nodiscard]] vstd::MD5 vk_semantic_header_md5(const VkShaderSerHeader &h) {
    auto bytes = CanonicalBytes{};
    bytes.u64(kVkComputeHeaderDomainTag);
    bytes.u64(h.header_ver);
    bytes.u32(h.pipeline_ver);
    bytes.md5(h.md5);
    bytes.md5(h.type_md5);
    bytes.md5(h.property_md5);
    bytes.md5(h.argument_md5);
    bytes.md5(h.spv_md5);
    bytes.u64(h.property_size);
    bytes.u64(h.spv_byte_size);
    for (auto size : h.block_size) { bytes.u32(size); }
    bytes.u32(h.kernel_arg_count);
    bytes.u32(h.printer_count);
    bytes.u32(h.printer_size_bytes);
    bytes.md5(h.printer_md5);
    bytes.u32(h.validation_count);
    bytes.u32(h.required_subgroup_size);
    bytes.u64(h.constant_ubo_size);
    bytes.md5(h.constant_ubo_md5);
    bytes.u8(h.use_bindless_buffer);
    bytes.u8(h.use_bindless_tex2d);
    bytes.u8(h.use_bindless_tex3d);
    bytes.u8(h.codegen_dialect);
    bytes.u64(h.required_spirv_features);
    LUISA_ASSERT(bytes.size() == 200u, "Vulkan builtin semantic header must be 200 bytes, got {}.", bytes.size());
    return bytes.digest();
}
void write_bytes(const std::filesystem::path &path, std::vector<std::byte> bytes) {
    if (auto parent = path.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream f{path, std::ios::binary | std::ios::trunc};
    if (!f) {
        LUISA_ERROR("Failed to open the output file: {}", path.string());
    }
    if (!bytes.empty()) {
        f.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    f.close();
    if (!f) {
        LUISA_ERROR("Failed to write the output file: {}", path.string());
    }
}
void append_bytes(std::vector<std::byte> &dst, const void *src, size_t size) {
    auto *p = reinterpret_cast<std::byte const *>(src);
    dst.insert(dst.end(), p, p + size);
}
template<typename T>
void append_value(std::vector<std::byte> &dst, const T &value) {
    append_bytes(dst, &value, sizeof(T));
}
/**
 * \brief Assemble the DX `ShaderSerializer` v5 artifact.
 */
[[nodiscard]] std::vector<std::byte> encode_dx_artifact(
    const Options &opt,
    std::string_view source,
    std::span<const hlsl::Property> properties,
    std::span<const std::byte> bytecode) {
#ifdef _WIN32
    auto root_sig = serialize_root_signature(properties);
    auto header = DxShaderSerHeader{};
    header.header_version = kDxHeaderVersion;
    header.md5 = md5_of(source);
    header.type_md5 = vstd::MD5{};
    header.root_sig_bytes = root_sig.size();
    header.code_bytes = bytecode.size();
    for (size_t i = 0u; i < 3u; ++i) {
        header.block_size[i] = opt.block_size[i];
    }
    header.property_count = static_cast<uint32_t>(properties.size());
    header.bindless_count = 0u;
    header.kernel_arg_count = 0u;
    header.printer_count = 0u;
    header.validation_count = 0u;
    auto result = std::vector<std::byte>{};
    result.reserve(sizeof(header) + root_sig.size() + bytecode.size() + properties.size_bytes());
    append_value(result, header);
    append_bytes(result, root_sig.data(), root_sig.size());
    append_bytes(result, bytecode.data(), bytecode.size());
    append_bytes(result, properties.data(), properties.size_bytes());
    return result;
#else
    // The DX artifact requires d3d12 root-signature serialization; rejected in
    // parse_options() on non-Windows hosts.
    return {};
#endif
}
/**
 * \brief Assemble the VK `ShaderSerializer` v10 compute artifact.
 */
[[nodiscard]] std::vector<std::byte> encode_vk_artifact(
    const Options &opt,
    std::string_view source,
    std::span<const hlsl::Property> properties,
    std::span<const std::byte> spirv) {
    auto header = VkShaderSerHeader{};
    header.header_ver = kVkShaderSerVersion;
    header.pipeline_ver = kVkXirPipelineVersion;
    header.md5 = md5_of(source);
    header.type_md5 = vstd::MD5{};
    header.property_size = properties.size();
    header.spv_byte_size = spirv.size();
    for (size_t i = 0u; i < 3u; ++i) {
        header.block_size[i] = opt.block_size[i];
    }
    header.kernel_arg_count = 0u;
    header.printer_count = 0u;
    header.printer_size_bytes = 0u;
    header.printer_md5 = md5_empty();
    header.validation_count = 0u;
    header.required_subgroup_size = 0u;
    header.constant_ubo_size = 0u;
    header.constant_ubo_md5 = md5_empty();
    header.use_bindless_buffer = 0u;
    header.use_bindless_tex2d = 0u;
    header.use_bindless_tex3d = 0u;
    header.codegen_dialect = kVkDialectHlslSpirv;
    header.required_spirv_features = 0u;
    header.property_md5 = vk_property_md5(properties);
    header.argument_md5 = md5_empty();
    header.spv_md5 = md5_of(spirv.data(), spirv.size());
    header.semantic_header_md5 = vk_semantic_header_md5(header);
    auto result = std::vector<std::byte>{};
    result.reserve(sizeof(header) + properties.size_bytes() + spirv.size());
    append_value(result, header);
    append_bytes(result, properties.data(), properties.size_bytes());
    append_bytes(result, spirv.data(), spirv.size());
    return result;
}
}// namespace
int main(int argc, char *argv[]) {
    Options opt;
    if (!parse_options(argc, argv, opt)) {
        return EXIT_FAILURE;
    }
    auto source = read_file(opt.input);
    if (source.empty()) {
        LUISA_ERROR("Input HLSL file is empty: {}", opt.input.string());
    }
    auto source_view = std::string_view{source};
    // The interface is derived from the source itself, so the emitted container
    // matches the hand-written one in `BuiltinKernel`.
    auto bindings = parse_bindings(source_view);
    if (bindings.empty()) {
        LUISA_ERROR("No :register(...) bindings found in {}", opt.input.string());
    }
    std::vector<hlsl::Property> properties;
    properties.reserve(bindings.size());
    for (auto &&b : bindings) {
        properties.emplace_back(hlsl::Property{b.type, b.space, b.register_index, b.array_size});
    }
    if (!opt.has_block_size) {
        uint32_t block[3]{1u, 1u, 1u};
        if (parse_block_size(source_view, block)) {
            std::memcpy(opt.block_size, block, sizeof(block));
        } else {
            LUISA_WARNING(
                "Cannot read a literal [numthreads(...)] from {} - using block size ({}, {}, {}). "
                "Pass --block-size to override.",
                opt.input.string(), opt.block_size[0], opt.block_size[1], opt.block_size[2]);
        }
    }
    // Compile with the very same dxc path the backends use.
    if (opt.shader_model < 10u) {
        LUISA_ERROR("Illegal shader model: {}", opt.shader_model);
    }
    const char *exe = (argc > 0 && argv[0]) ? argv[0] : "";
    Context context{exe};
    hlsl::ShaderCompiler compiler{context.runtime_directory(), opt.is_spirv};
    auto dxc_args = dxc_arguments(opt);
    auto wide_args = std::vector<LPCWSTR>{};
    wide_args.reserve(dxc_args.size());
    for (auto &&a : dxc_args) { wide_args.emplace_back(a.c_str()); }
    auto code = compiler.compile(source_view, wide_args);
    auto bytecode = code.multi_visit_or(
        std::vector<std::byte>{},
        [&](hlsl::ComUniquePtr<IDxcBlob> &blob) {
            auto *begin = reinterpret_cast<std::byte const *>(blob->GetBufferPointer());
            return std::vector<std::byte>{begin, begin + blob->GetBufferSize()};
        },
        [&](auto &&error) {
            LUISA_ERROR("DXC compile error for builtin '{}': {}", opt.name, std::string_view{error});
            return std::vector<std::byte>{};
        });
    if (bytecode.empty()) {
        return EXIT_FAILURE;
    }
    auto artifact = opt.raw
                        ? std::move(bytecode)
                        : (opt.is_spirv
                               ? encode_vk_artifact(opt, source_view, properties, bytecode)
                               : encode_dx_artifact(opt, source_view, properties, bytecode));
    if (artifact.empty()) {
        LUISA_ERROR("Failed to encode the builtin artifact for '{}'.", opt.name);
        return EXIT_FAILURE;
    }
    write_bytes(opt.output, std::move(artifact));
    if (!std::filesystem::is_regular_file(opt.output)) {
        LUISA_ERROR("Bytecode was not written to {}", opt.output.string());
        return EXIT_FAILURE;
    }
    LUISA_INFO("Builtin '{}' ({}{}): {} properties, block ({}, {}, {}), {} bytes -> {}",
               opt.name, opt.is_spirv ? "vk" : "dx", opt.raw ? ", raw" : "",
               properties.size(),
               opt.block_size[0], opt.block_size[1], opt.block_size[2],
               std::filesystem::file_size(opt.output),
               opt.output.string());
    for (auto &&property : properties) {
        LUISA_INFO("  property: type={} space={} register={} array_size={}",
                   static_cast<int>(property.type), property.space_index,
                   property.register_index, property.array_size);
    }
    return EXIT_SUCCESS;
}
