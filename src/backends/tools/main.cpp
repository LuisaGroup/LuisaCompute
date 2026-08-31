// lc-compile-builtin: AOT-compile a shader source into the bytecode containers
// read by the backend builtin-kernel loaders, and (for the Vulkan backend) into
// the embedded device-library arrays the build system generates.
//
// `src/backends/dx/Shader/BuiltinKernel.cpp` and `src/backends/vk/builtin_kernel.cpp`
// pick their helper shaders up either as embedded blobs (the
// `src/backends/common/hlsl/builtin/*.dxil` files, or the generated
// `vulkan_builtin_spirv_embedded.{h,cpp}` arrays) or through the shader
// serializers. This tool regenerates those artifacts from the canonical sources
// with the exact same compiler the runtime / the backend build uses:
//
//   * `dx`  - compiles raw HLSL with dxc and writes the DX `ShaderSerializer` v5
//     artifact (header + serialized root signature + DXBC + properties), which
//     `ComputeShader::compile_compute(..., CacheType::Internal, ...)` loads.
//   * `vk`  - compiles raw HLSL with dxc `-spirv` (the legacy HLSL_SPIRV route)
//     and writes the VK `ShaderSerializer` v10 compute artifact
//     (header + properties + SPIR-V), i.e. `SerdeType::kBuiltin` bytecode.
//   * `spv` - compiles the *new* Vulkan builtin sources
//     (`src/backends/vk/builtin/*.comp.hlsl`) with glslang exactly like the
//     backend build does (`luisa-glslang -D -V --target-env vulkan1.2 -S comp
//     -e main`), validates the module with `luisa-validate-spirv`, and writes
//     the bare SPIR-V module. `--container` wraps it into a VK v10 artifact with
//     the `VULKAN_BUILTIN` codegen dialect instead; `--embed` additionally emits
//     the `luisa_compute_vk_builtin_*` device-library arrays.
//   * `inspect`  - decodes an existing artifact and prints its contract fields
//     (versions, digests, block size, properties, dialect, section sizes).
//   * `embed`    - wraps finished SPIR-V modules into the generated
//     `vulkan_builtin_spirv_embedded.{h,cpp}` device-library pair.
//
// The resource properties - and with them the root-signature/descriptor layout -
// are parsed from the `:register(...)` annotations of the input source, and the
// block size from its `[numthreads(...)]` attribute (macro-valued attributes are
// resolved against the `*.def` layout contracts the sources mirror), so the
// emitted interface matches the hand-written tables in `BuiltinKernel`.
//
// Verification (`--verify`, `inspect`) re-reads the artifact through
// `luisa::BinaryIO`/`luisa::BinaryStream` - the very runtime interface the
// backends read builtin blobs through - so an artifact this tool accepts is an
// artifact a backend can load.
//
// Usage:
//   lc-compile-builtin <dx|vk|spv> <input-shader> <output> [options]
//   lc-compile-builtin <dx|vk> inspect <artifact> [--name <label>]
//   lc-compile-builtin spv embed <module.spv>... -o <embedded.cpp> [-h <embedded.h>]
// e.g.:
//   lc-compile-builtin dx src/backends/common/hlsl/builtin/bindless_upload.bytes \
//                       src/backends/common/hlsl/builtin/load_bdls.dxil
//   lc-compile-builtin vk src/backends/common/hlsl/builtin/bindless_upload.bytes \
//                       src/backends/common/hlsl/builtin/load_bdls_vk.dxil
//   lc-compile-builtin spv src/backends/vk/builtin/accel_process.comp.hlsl \
//                          accel_process.spv
//   lc-compile-builtin spv src/backends/vk/builtin/accel_process.comp.hlsl \
//                          accel_process.dxil --contract accel_process --container
//
// Options:
//   --entry <name>         compiler entry point (default: main)
//   --shader-model <n>     packed shader model, e.g. 62 for cs_6_2 (default: 62)
//   --block-size <x[,y,z]> override the parsed [numthreads(...)] block size
//   --include <dir>        extra source include directory (repeatable)
//   --prepend <file>       text-prepend a header/snippet before the input shader
//                          (BC builtins: bc6_header / bc7_header, exactly like the
//                          runtime `BuiltinKernel::_load_bc_kernel` does)
//   --define <MACRO=VALUE> glslang preprocessor definition (repeatable)
//   --contract <name>      Vulkan builtin contract: indirect_prepare |
//                          accel_process | bindless_upload
//   --sampler-heap         append the canonical SamplerHeap property
//   --push-constant <n>    append a push-constant property of <n> bytes (vk)
//   --container            (spv) write a VK v10 artifact instead of raw SPIR-V
//   --no-optimize          compile with dxc -Od instead of -O3
//   --raw                  write the bare compiler blob instead of an artifact
//   --embed                (spv) also write the embedded device-library pair
//   --install              also store the artifact in the runtime data/cache dir
//   --store <cache|data>   which runtime store --install writes (default: data)
//   --verify               re-read the artifact through luisa::BinaryIO and
//                          validate the container contract
//   --target-env <env>     glslang SPIR-V target environment (default: vulkan1.2)
//   --glslang <exe>        luisa-glslang host tool
//   --validator <exe>      luisa-validate-spirv host tool
//   --embedder <exe>       luisa-embed-device-lib host tool
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
#include <luisa/core/binary_file_stream.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/runtime/context.h>
#include <luisa/vstl/md5.h>
#include "../common/hlsl/shader_compiler.h"
#include "../common/hlsl/shader_property.h"
#include "../common/subprocess.h"
#include <reproc++/run.hpp>
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
 * \brief Which artifact a compiled blob is packaged into.
 */
enum class Target : uint8_t {
    DX,            // DX ShaderSerializer v5 container (dxc, DXBC payload)
    VK_HLSL_SPIRV, // VK v10 container, dialect HLSL_SPIRV (dxc -spirv)
    VK_BUILTIN     // glslang route: raw SPIR-V, or a VK v10 container with the
                   // VULKAN_BUILTIN dialect when --container is given
};
/**
 * \brief What the invocation asks the tool to do.
 */
enum class Action : uint8_t {
    COMPILE,
    INSPECT,
    EMBED
};
/**
 * \brief Compile request parsed from the command line.
 */
struct Options {
    Action action{Action::COMPILE};
    Target target{Target::DX};
    std::filesystem::path input;
    std::filesystem::path prepend;
    std::filesystem::path output;
    std::vector<std::filesystem::path> inputs;
    std::filesystem::path embedded_source;
    std::filesystem::path embedded_header;
    std::string name{kDefaultKernelName};
    std::string entry{"main"};
    std::string target_env{"vulkan1.2"};
    uint32_t shader_model{62u};
    uint32_t block_size[3]{1u, 1u, 1u};
    bool has_block_size{false};
    std::vector<std::string> includes;
    std::vector<std::string> defines;
    std::string contract;
    bool sampler_heap{false};
    bool container{false};
    int32_t push_constant_bytes{-1};
    bool optimize{true};
    bool raw{false};
    bool embed{false};
    bool install{false};
    bool verify{false};
    bool store_cache{false};
    std::filesystem::path glslang;
    std::filesystem::path validator;
    std::filesystem::path embedder;
    /**
     * \brief True when dxc must emit SPIR-V (`vk`).
     */
    [[nodiscard]] bool is_spirv() const noexcept {
        return target == Target::VK_HLSL_SPIRV;
    }
    /**
     * \brief True when the packaged container is the VK v10 one.
     */
    [[nodiscard]] bool is_vk_container() const noexcept {
        return target == Target::VK_HLSL_SPIRV || (target == Target::VK_BUILTIN && container && !raw);
    }
};
[[nodiscard]] std::filesystem::path absolute_path(std::string_view p) {
    auto path = std::filesystem::path{p};
    return path.is_absolute() ? path : std::filesystem::absolute(path);
}
void print_usage(const char *exe) {
    LUISA_INFO(
        "Usage: {} <dx|vk|spv> <input-shader> <output> [options]\n"
        "       {} <dx|vk> inspect <artifact> [--name <label>]\n"
        "       {} spv embed <module.spv>... -o <embedded.cpp> [-h <embedded.h>]\n"
        "  dx|vk|spv            target bytecode dialect\n"
        "                         dx  - DX v5 container, dxc -> DXBC (BuiltinKernel .dxil)\n"
        "                         vk  - VK v10 container, dxc -spirv (HLSL_SPIRV dialect)\n"
        "                         spv - glslang route (VULKAN_BUILTIN dialect):\n"
        "                               src/backends/vk/builtin/*.comp.hlsl -> .spv\n"
        "  input-shader         raw HLSL builtin source, e.g.\n"
        "                         src/backends/common/hlsl/builtin/bindless_upload.bytes\n"
        "                         src/backends/vk/builtin/accel_process.comp.hlsl\n"
        "  output               destination path (load_bdls.dxil, accel_process.spv, ...)\n"
        "  --entry <name>         compiler entry point (default: main)\n"
        "  --shader-model <n>     packed shader model, 62 == cs_6_2 (default: 62)\n"
        "  --block-size <x[,y,z]> override [numthreads(...)]\n"
        "  --include <dir>        extra include directory (repeatable)\n"
        "  --define <MACRO=VAL>   glslang preprocessor definition (repeatable)\n"
        "  --prepend <file>       text-prepend a header/snippet before the input shader\n"
        "                           (BC builtins: bc6_header / bc7_header, like the runtime\n"
        "                           `BuiltinKernel::_load_bc_kernel` does)\n"
        "  --contract <name>      Vulkan builtin contract:\n"
        "                           indirect_prepare | accel_process | bindless_upload\n"
        "  --sampler-heap         append the canonical SamplerHeap property\n"
        "  --push-constant <n>    append an n-byte push-constant property\n"
        "  --container            (spv) write a VK v10 artifact, not raw SPIR-V\n"
        "  --no-optimize          compile with -Od instead of -O3\n"
        "  --raw                  write the bare compiler blob instead of an artifact\n"
        "  --embed                (spv) also write the embedded device-library pair\n"
        "  --install              also store the artifact in the runtime data/cache directory\n"
        "  --store <cache|data>   runtime store written by --install (default: data)\n"
        "  --verify               re-read the artifact through luisa::BinaryIO and validate it\n"
        "  --target-env <env>     glslang SPIR-V target env (default: vulkan1.2)\n"
        "  --glslang <exe>        luisa-glslang host tool (default: <runtime dir>)\n"
        "  --validator <exe>      luisa-validate-spirv host tool (default: <runtime dir>)\n"
        "  --embedder <exe>       luisa-embed-device-lib host tool (default: <runtime dir>)\n"
        "  --name <label>         shader label used in log messages",
        exe, exe, exe);
}
/**
 * \brief Parse the command line. Returns false (without throwing) on usage errors.
 */
[[nodiscard]] bool parse_options(int argc, char *argv[], Options &opt) noexcept {
    // Usage problems are reported without aborting so the caller gets exit code 1.
    auto usage_error = [&](std::string_view message) noexcept {
        LUISA_WARNING("{}", message);
    };
    auto parse_block_size_argument = [&](std::string_view value) noexcept {
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
    };
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        auto arg = std::string_view{argv[i]};
        if (arg == "--help") {
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
        if (arg == "--embed") {
            opt.embed = true;
            continue;
        }
        if (arg == "--container") {
            opt.container = true;
            continue;
        }
        if (arg == "--install") {
            opt.install = true;
            continue;
        }
        if (arg == "--verify") {
            opt.verify = true;
            continue;
        }
        if (arg == "--sampler-heap") {
            opt.sampler_heap = true;
            continue;
        }
        if (arg.front() != '-') {
            positional.emplace_back(arg);
            continue;
        }
        // -o/-h keep the `embed_device_lib` spelling for the embed action and the
        // device-library output pair.
        if (arg == "-o") {
            if (i + 1 >= argc) {
                usage_error("option -o requires a value.");
                return false;
            }
            opt.embedded_source = absolute_path(argv[++i]);
            continue;
        }
        if (arg == "-h") {
            if (i + 1 >= argc) {
                usage_error("option -h requires a value.");
                return false;
            }
            opt.embedded_header = absolute_path(argv[++i]);
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
        } else if (arg == "--contract") {
            opt.contract = std::string{value};
        } else if (arg == "--target-env") {
            opt.target_env = std::string{value};
        } else if (arg == "--include") {
            opt.includes.emplace_back(value);
        } else if (arg == "--prepend") {
            opt.prepend = absolute_path(value);
        } else if (arg == "--define") {
            opt.defines.emplace_back(value);
        } else if (arg == "--shader-model") {
            opt.shader_model = static_cast<uint32_t>(std::strtoul(value.data(), nullptr, 10));
        } else if (arg == "--block-size") {
            parse_block_size_argument(value);
        } else if (arg == "--push-constant") {
            opt.push_constant_bytes = static_cast<int32_t>(std::strtol(value.data(), nullptr, 10));
        } else if (arg == "--store") {
            if (value == "cache") {
                opt.store_cache = true;
            } else if (value == "data") {
                opt.store_cache = false;
            } else {
                usage_error("--store expects 'cache' or 'data'.");
                return false;
            }
        } else if (arg == "--glslang") {
            opt.glslang = absolute_path(value);
        } else if (arg == "--validator") {
            opt.validator = absolute_path(value);
        } else if (arg == "--embedder") {
            opt.embedder = absolute_path(value);
        } else {
            usage_error("unknown option: " + std::string{arg});
            return false;
        }
    }
    if (positional.empty()) {
        print_usage(argc > 0 ? argv[0] : "lc-compile-builtin");
        return false;
    }
    // <dx|vk|spv> [inspect|embed] <inputs...> <output>
    auto backend = std::string_view{positional[0]};
    auto arguments = std::vector<std::string>{positional.begin() + 1, positional.end()};
    if (backend == "dx") {
        opt.target = Target::DX;
    } else if (backend == "vk") {
        opt.target = Target::VK_HLSL_SPIRV;
    } else if (backend == "spv" || backend == "vk-builtin") {
        opt.target = Target::VK_BUILTIN;
    } else {
        usage_error("unknown backend '" + std::string{backend} + "': expected 'dx', 'vk' or 'spv'.");
        return false;
    }
#ifndef _WIN32
    if (opt.target == Target::DX) {
        usage_error("the DX builtin artifact requires a Windows host (d3d12 root-signature serialization).");
        return false;
    }
#endif
    if (!arguments.empty() && arguments[0] == "inspect") {
        arguments.erase(arguments.begin());
        opt.action = Action::INSPECT;
    } else if (!arguments.empty() && arguments[0] == "embed") {
        arguments.erase(arguments.begin());
        opt.action = Action::EMBED;
    }
    if (opt.action == Action::INSPECT) {
        if (opt.target == Target::VK_BUILTIN) {
            usage_error("inspect requires 'dx' or 'vk': a raw .spv module has no container to decode.");
            return false;
        }
        if (arguments.size() != 1u) {
            usage_error("inspect takes exactly one artifact path.");
            return false;
        }
        if (opt.input = absolute_path(arguments[0]); !std::filesystem::is_regular_file(opt.input)) {
            usage_error("artifact file not found: " + opt.input.string());
            return false;
        }
        return true;
    }
    if (opt.action == Action::EMBED) {
        if (opt.target != Target::VK_BUILTIN) {
            usage_error("embed is a Vulkan-builtin operation; use 'spv embed'.");
            return false;
        }
        if (arguments.empty()) {
            usage_error("embed takes at least one SPIR-V module.");
            return false;
        }
        opt.inputs.reserve(arguments.size());
        for (auto &&argument : arguments) {
            auto path = absolute_path(argument);
            if (!std::filesystem::is_regular_file(path)) {
                usage_error("SPIR-V module not found: " + path.string());
                return false;
            }
            opt.inputs.emplace_back(std::move(path));
        }
        if (opt.embedded_source.empty()) {
            usage_error("embed requires -o <embedded.cpp>.");
            return false;
        }
        return true;
    }
    if (arguments.size() != 2u) {
        print_usage(argc > 0 ? argv[0] : "lc-compile-builtin");
        return false;
    }
    if (opt.target != Target::VK_BUILTIN && (opt.embed || opt.container)) {
        usage_error("--embed/--container apply to the 'spv' (Vulkan builtin) target only.");
        return false;
    }
    if (opt.input = absolute_path(arguments[0]); !std::filesystem::is_regular_file(opt.input)) {
        usage_error("input shader file not found: " + opt.input.string());
        return false;
    }
    if (!opt.prepend.empty() && !std::filesystem::is_regular_file(opt.prepend)) {
        usage_error("--prepend file not found: " + opt.prepend.string());
        return false;
    }
    opt.output = absolute_path(arguments[1]);
    return true;
}
[[nodiscard]] std::string read_file(const std::filesystem::path &path) {
    std::ifstream f{path, std::ios::binary};
    if (!f) {
        LUISA_ERROR("Failed to open input shader file: {}", path.string());
    }
    return std::string{std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
}
/**
 * \brief Read a whole file as bytes.
 */
[[nodiscard]] std::vector<std::byte> read_bytes(const std::filesystem::path &path) {
    std::ifstream f{path, std::ios::binary | std::ios::ate};
    if (!f) {
        LUISA_ERROR("Failed to open {}.", path.string());
    }
    auto size = static_cast<size_t>(f.tellg());
    auto bytes = std::vector<std::byte>(size);
    f.seekg(0);
    if (size > 0u) {
        f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    }
    return bytes;
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
 * \brief Collect `static const uint NAME = <literal>;` declarations and object-like
 * `#define NAME <literal>` macros.
 *
 * The Vulkan builtin sources mirror the `*.def` layout contracts into
 * `static const uint` values (`LC_INDIRECT_PREPARE_BLOCK_SIZE`,
 * `LC_VULKAN_ACCEL_BLOCK_SIZE`) and `#define` macros
 * (`THREAD_GROUP_SIZE` in the BC headers), so a `[numthreads(...)]` that is not
 * a literal can still be resolved without running a preprocessor.
 */
[[nodiscard]] std::vector<std::pair<std::string, uint32_t>> parse_scalar_consts(std::string_view src) {
    std::vector<std::pair<std::string, uint32_t>> constants;
    size_t search_from{};
    while (search_from < src.size()) {
        auto mark = src.find("#define", search_from);
        if (mark != std::string_view::npos) {
            auto p = mark + 7u;
            search_from = p;
            while (p < src.size() && is_space(src[p])) { ++p; }
            auto name_begin = p;
            while (p < src.size() && is_ident_char(src[p])) { ++p; }
            if (p > name_begin) {
                auto name = std::string{src.substr(name_begin, p - name_begin)};
                auto value_begin = p;
                while (p < src.size() && is_space(src[p])) { ++p; }
                if (p < src.size() && src[p] == '(') {
                    // Object-like only; skip function-like macros.
                    search_from = p + 1u;
                    continue;
                }
                auto digits = p;
                while (p < src.size() && is_digit(src[p])) { ++p; }
                if (p > digits && p < src.size() && (is_space(src[p]) || src[p] == '\r' || src[p] == '\n')) {
                    uint32_t value{};
                    for (auto q = digits; q < p; ++q) {
                        value = value * 10u + static_cast<uint32_t>(src[q] - '0');
                    }
                    constants.emplace_back(std::move(name), value);
                }
            }
            continue;
        }
        auto static_mark = src.find("static const uint", search_from);
        if (static_mark == std::string_view::npos) { break; }
        auto p = static_mark + 17u;
        search_from = p;
        while (p < src.size() && is_space(src[p])) { ++p; }
        auto name_begin = p;
        while (p < src.size() && is_ident_char(src[p])) { ++p; }
        if (p == name_begin) { continue; }
        auto name = std::string{src.substr(name_begin, p - name_begin)};
        while (p < src.size() && is_space(src[p])) { ++p; }
        if (p >= src.size() || src[p] != '=') { continue; }
        ++p;
        while (p < src.size() && is_space(src[p])) { ++p; }
        if (p >= src.size() || !is_digit(src[p])) { continue; }
        uint32_t value{};
        while (p < src.size() && is_digit(src[p])) {
            value = value * 10u + static_cast<uint32_t>(src[p++] - '0');
        }
        constants.emplace_back(std::move(name), value);
    }
    return constants;
}
/**
 * \brief Resolve one `[numthreads(...)]` operand: a literal, or a name mirrored
 * by the `*.def` layout contracts.
 */
[[nodiscard]] bool eval_block_size_operand(
    std::string_view token,
    std::span<const std::pair<std::string, uint32_t>> constants,
    uint32_t &value) noexcept {
    auto trimmed = token;
    while (!trimmed.empty() && is_space(trimmed.front())) { trimmed.remove_prefix(1u); }
    while (!trimmed.empty() && is_space(trimmed.back())) { trimmed.remove_suffix(1u); }
    size_t digits{};
    while (digits < trimmed.size() && is_digit(trimmed[digits])) { ++digits; }
    if (digits == trimmed.size() && digits > 0u) {
        value = 0u;
        for (size_t i = 0u; i < digits; ++i) {
            value = value * 10u + static_cast<uint32_t>(trimmed[i] - '0');
        }
        return true;
    }
    // A trailing `u` suffix is part of the literal.
    if (digits + 1u == trimmed.size() && trimmed[digits] == 'u') {
        value = 0u;
        for (size_t i = 0u; i < digits; ++i) {
            value = value * 10u + static_cast<uint32_t>(trimmed[i] - '0');
        }
        return true;
    }
    auto name = std::string{trimmed};
    auto it = std::find_if(constants.begin(), constants.end(),
                           [&name](auto &&entry) noexcept { return entry.first == name; });
    if (it == constants.end()) { return false; }
    value = it->second;
    return true;
}
/**
 * \brief Parse `[numthreads(x, y, z)]`, resolving layout-contract macros.
 * Returns false when an operand is neither a literal nor a mirrored constant.
 */
[[nodiscard]] bool parse_block_size(
    std::string_view src,
    std::span<const std::pair<std::string, uint32_t>> constants,
    uint32_t block[3]) noexcept {
    auto pos = src.find("numthreads(");
    if (pos == std::string_view::npos) { return false; }
    pos += 11u;
    for (uint32_t i = 0u; i < 3u; ++i) {
        auto token_begin = pos;
        while (pos < src.size() && src[pos] != ',' && src[pos] != ')') { ++pos; }
        if (!eval_block_size_operand(src.substr(token_begin, pos - token_begin), constants, block[i])) {
            return false;
        }
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
    if (opt.is_spirv()) {
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
    // The builtin sources `#include` their layout contracts
    // (indirect_dispatch_layout.def, vulkan_accel_update_layout.def); the runtime
    // reaches them through the embedded file system, so an AOT compile needs the
    // same search paths explicitly.
    for (auto &&dir : opt.includes) {
        args.emplace_back(L"-I" + std::wstring{dir.begin(), dir.end()});
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
// vk::detail::ShaderCodegenDialect - the runtime rejects an artifact whose
// dialect does not match the loader that reads it.
constexpr uint8_t kVkDialectHlslSpirv = 0u;
constexpr uint8_t kVkDialectVulkanBuiltin = 3u;
// vk::detail::descriptor_interface_sampler_count - the immutable sampler heap
// every Vulkan shader interface must declare exactly once.
constexpr uint32_t kVulkanSamplerCount = 16u;
constexpr uint32_t kSpirvMagic = 0x07230203u;
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
    LUISA_ASSERT(bytes.size() == 200u, "Vulkan shader semantic-header encoding must produce 200 bytes, got {}.", bytes.size());
    return bytes.digest();
}
void write_bytes(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    if (auto parent = path.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    if (bytes.empty()) { return; }
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
void write_text(const std::filesystem::path &path, std::string_view text) {
    write_bytes(path, std::span<const std::byte>{reinterpret_cast<std::byte const *>(text.data()), text.size()});
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
 * \brief Append the property table field by field.
 *
 * `hlsl::Property` is `{uint8_t type, uint space, uint register, uint count}`,
 * so each record carries three padding bytes. Copying the struct verbatim would
 * leak indeterminate padding into the container and make two compiles of the
 * same source differ, which defeats the whole point of an AOT artifact. Writing
 * the fields at their exact offsets keeps the layout bit-identical to what
 * `ShaderSerializer` produces while making the output reproducible.
 */
void append_properties(std::vector<std::byte> &dst, std::span<const hlsl::Property> properties) {
    for (auto &&property : properties) {
        append_bytes(dst, &property.type, sizeof(property.type));
        auto pad = std::array<std::byte, 3>{};
        append_bytes(dst, pad.data(), pad.size());
        append_bytes(dst, &property.space_index, sizeof(property.space_index));
        append_bytes(dst, &property.register_index, sizeof(property.register_index));
        append_bytes(dst, &property.array_size, sizeof(property.array_size));
    }
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
    append_properties(result, properties);
    return result;
#else
    // The DX artifact requires d3d12 root-signature serialization; rejected in
    // parse_options() on non-Windows hosts.
    return {};
#endif
}
/**
 * \brief Assemble the VK `ShaderSerializer` v10 compute artifact.
 *
 * The section order and the digest recipe follow
 * `vk::detail::encode_compute_shader_artifact` so that the runtime loader
 * (`decode_compute_shader_artifact`) accepts the result: `property_size` is a
 * *count*, `spv_byte_size` a byte size, and both the codegen dialect and the
 * SPIR-V feature mask are validated on load.
 */
[[nodiscard]] std::vector<std::byte> encode_vk_artifact(
    const Options &opt,
    std::string_view source,
    std::span<const hlsl::Property> properties,
    std::span<const std::byte> spirv) {
    LUISA_ASSERT(!spirv.empty() && spirv.size() % sizeof(uint32_t) == 0u,
                 "A Vulkan artifact needs a word-aligned SPIR-V payload, got {} bytes.",
                 spirv.size());
    auto header = VkShaderSerHeader{};
    header.header_ver = kVkShaderSerVersion;
    header.pipeline_ver = kVkXirPipelineVersion;
    header.md5 = md5_of(source);
    header.type_md5 = vstd::MD5{};
    header.property_size = properties.size();// a property count, not a byte size
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
    // The codegen dialect selects the loader contract: DXC-produced SPIR-V is
    // HLSL_SPIRV, while the glslang-built backend-private kernels are
    // VULKAN_BUILTIN (see vk::detail::ShaderCodegenDialect).
    header.codegen_dialect = opt.target == Target::VK_BUILTIN ? kVkDialectVulkanBuiltin : kVkDialectHlslSpirv;
    // Only the native XIR dialect is capability-reconciled by the codec, so a
    // builtin never claims a device feature.
    header.required_spirv_features = 0u;
    header.property_md5 = vk_property_md5(properties);
    header.argument_md5 = md5_empty();
    header.spv_md5 = md5_of(spirv.data(), spirv.size());
    header.semantic_header_md5 = vk_semantic_header_md5(header);
    auto result = std::vector<std::byte>{};
    result.reserve(sizeof(header) + properties.size_bytes() + spirv.size());
    append_value(result, header);
    append_properties(result, properties);
    append_bytes(result, spirv.data(), spirv.size());
    return result;
}
/**
 * \brief The `detail::vulkan_builtin_buffer_properties` table the Vulkan builtin
 * loader attaches to every backend-private kernel, and the `[numthreads(...)]`
 * side of `detail::vulkan_builtin_kernel_contract`.
 */
[[nodiscard]] bool vk_builtin_contract(std::string_view contract,
                                       std::vector<hlsl::Property> &properties,
                                       uint32_t block[3]) noexcept {
    if (contract == "indirect_prepare") {
        block[0] = 64u;
    } else if (contract == "accel_process" || contract == "bindless_upload") {
        block[0] = 256u;
    } else {
        return false;
    }
    block[1] = 1u;
    block[2] = 1u;
    properties = {
        hlsl::Property{hlsl::ShaderVariableType::StructuredBuffer, 0u, 0u, 1u},
        hlsl::Property{hlsl::ShaderVariableType::RWStructuredBuffer, 0u, 1u, 1u},
        hlsl::Property{hlsl::ShaderVariableType::SamplerHeap, 1u, 0u, kVulkanSamplerCount},
    };
    return true;
}
/**
 * \brief A `luisa::BinaryIO` over a directory pair, mirroring the file-backed
 * branch of `DefaultBinaryIO`. Verification reads an artifact back through the
 * same virtual interface (`read_internal_shader`/`read_shader_cache`) the
 * backends' `read_binary_io` dispatches to.
 */
class DirectoryBinaryIO final : public BinaryIO {
public:
    DirectoryBinaryIO(std::filesystem::path data_dir, std::filesystem::path cache_dir) noexcept
        : _data_dir{std::move(data_dir)}, _cache_dir{std::move(cache_dir)} {}
    void clear_shader_cache() const noexcept override {
        std::error_code ec;
        std::filesystem::remove_all(_cache_dir, ec);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override {
        return open(name, _data_dir);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override {
        return open(name, _cache_dir);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override {
        return open(name, _data_dir);
    }
    luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override {
        return store(_data_dir, name, data);
    }
    luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override {
        return store(_cache_dir, name, data);
    }
    luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override {
        return store(_data_dir, name, data);
    }

private:
    [[nodiscard]] static std::filesystem::path resolve(const std::filesystem::path &dir, luisa::string_view name) {
        auto path = std::filesystem::path{name};
        return path.is_absolute() ? path : dir / path.filename();
    }
    [[nodiscard]] static luisa::unique_ptr<BinaryStream> open(luisa::string_view name, const std::filesystem::path &dir) {
        auto path = resolve(dir, name);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) { return {}; }
        if (auto *file = std::fopen(path.string().c_str(), "rb")) [[likely]] {
                          auto length = luisa::detail::get_c_file_length(file);
            if (length == 0u) [[unlikely]] {
                std::fclose(file);
                return {};
            }
            return luisa::make_unique<BinaryFileStream>(file, length);
        }
        return {};
    }
    [[nodiscard]] luisa::filesystem::path store(const std::filesystem::path &dir, luisa::string_view name, luisa::span<std::byte const> data) const {
        auto path = resolve(dir, name);
        write_bytes(path, data);
        return path;
    }
    std::filesystem::path _data_dir;
    std::filesystem::path _cache_dir;
};
/**
 * \brief Locate a build-tree host tool (`luisa-glslang`, `luisa-validate-spirv`,
 * `luisa-embed-device-lib`): it lives next to this executable, i.e. in the
 * runtime directory the `Context` resolves.
 */
[[nodiscard]] std::filesystem::path find_host_tool(const Context &context, std::string_view name) {
    auto file_name = std::string{name}
#ifdef _WIN32
                     + ".exe"
#endif
        ;
    if (auto candidate = context.runtime_directory() / file_name; std::filesystem::is_regular_file(candidate)) {
        return candidate;
    }
    LUISA_WARNING("Host tool '{}' not found in the runtime directory {}.", name, context.runtime_directory().string());
    return {};
}
/**
 * \brief Run a host tool and report its exit code plus merged stdout/stderr.
 */
struct ToolRunResult {
    int exit_code;
    std::string output;
    [[nodiscard]] bool ok() const noexcept { return exit_code == 0; }
};
[[nodiscard]] ToolRunResult run_tool(const std::filesystem::path &program, const std::vector<std::string> &arguments) {
    auto argv = std::vector<std::string>{};
    argv.reserve(arguments.size() + 1u);
    argv.emplace_back(program.string());
    for (auto &&argument : arguments) { argv.emplace_back(argument); }
    // Default redirection pipes stdout/stderr back here so a failing glslang or
    // SPIRV-Tools run can be reported with its diagnostics.
    auto options = reproc::options{};
    auto output = std::string{};
    auto [exit_code, error] = reproc::run(argv, options, reproc::sink::string{output}, reproc::sink::string{output});
    if (error) {
        LUISA_ERROR("Failed to run '{}': {}.", program.string(), error.message());
    }
    return {exit_code, std::move(output)};
}
/**
 * \brief The include search path for a shader source: its own directory (where a
 * mirrored `*.def` contract lives) plus the common backend directory.
 */
[[nodiscard]] std::vector<std::string> default_include_dirs(const std::filesystem::path &source) {
    auto dirs = std::vector<std::string>{};
    auto push = [&dirs](const std::filesystem::path &dir) {
        std::error_code ec;
        if (dir.empty() || !std::filesystem::is_directory(dir, ec)) { return; }
        auto key = dir.generic_string();
        if (std::find(dirs.begin(), dirs.end(), key) == dirs.end()) { dirs.emplace_back(key); }
    };
    push(source.parent_path());
    push(source.parent_path().parent_path() / "common");
    push(std::filesystem::current_path() / "src" / "backends" / "common");
    push(std::filesystem::path{__FILE__}.parent_path().parent_path() / "common");
    return dirs;
}
/**
 * \brief Compile a Vulkan builtin HLSL source with glslang, using the exact
 * command the Vulkan backend build issues (`src/backends/vk/xmake.lua`).
 */
[[nodiscard]] std::vector<std::byte> compile_with_glslang(
    const Context &context,
    const Options &opt) {
    auto glslang = opt.glslang.empty() ? find_host_tool(context, "luisa-glslang") : opt.glslang;
    LUISA_ASSERT(!glslang.empty(),
                 "The glslang route needs the luisa-glslang host tool: build it with "
                 "`xmake build lc-glslang-standalone` or pass --glslang <exe>.");
    // A scratch copy under the output directory lets any input spelling
    // (`*.bytes`, `*.comp.hlsl`) be reused verbatim.
    auto scratch_dir = opt.output.parent_path() / ".lc-compile-builtin";
    auto scratch_source = scratch_dir / (opt.name + ".comp.hlsl");
    write_bytes(scratch_source, read_bytes(opt.input));
    auto scratch_module = scratch_dir / (opt.name + ".spv");
    {
        std::error_code ec;
        std::filesystem::remove(scratch_module, ec);
    }
    auto arguments = std::vector<std::string>{
        "-D",// read HLSL
        "-V",
        "--target-env",
        opt.target_env.empty() ? "vulkan1.2" : opt.target_env,
        "-S",
        "comp",
        "-e",
        opt.entry.empty() ? "main" : opt.entry,
    };
    auto includes = opt.includes;
    if (includes.empty()) {
        includes = default_include_dirs(opt.input);
    }
    for (auto &&dir : includes) {
        // glslang requires the argument to immediately follow `-I`.
        arguments.emplace_back("-I" + dir);
    }
    for (auto &&definition : opt.defines) {
        arguments.emplace_back("-D" + definition);
    }
    arguments.emplace_back("-o");
    arguments.emplace_back(scratch_module.string());
    arguments.emplace_back(scratch_source.string());
    auto compiled = run_tool(glslang, arguments);
    if (!compiled.ok()) {
        LUISA_ERROR("glslang failed for builtin '{}' (exit {}):\n{}", opt.name, compiled.exit_code, compiled.output);
    }
    auto module = read_bytes(scratch_module);
    if (module.empty()) {
        LUISA_ERROR("glslang produced an empty SPIR-V module for builtin '{}'.", opt.name);
    }
    // SPIRV-Tools validation, exactly like `lc-vk-validate-spirv <out>.spv`.
    if (auto validator = opt.validator.empty() ? find_host_tool(context, "luisa-validate-spirv") : opt.validator;
        !validator.empty()) {
        auto validated = run_tool(validator, {scratch_module.string()});
        if (!validated.ok()) {
            LUISA_ERROR("SPIR-V validation failed for builtin '{}' (exit {}):\n{}",
                        opt.name, validated.exit_code, validated.output);
        }
    }
    {
        std::error_code ec;
        std::filesystem::remove(scratch_source, ec);
        std::filesystem::remove(scratch_module, ec);
        std::filesystem::remove(scratch_dir, ec);// only succeeds when empty
    }
    return module;
}
/**
 * \brief Emit the generated device-library pair for finished SPIR-V modules by
 * driving `luisa-embed-device-lib` with the arguments the VK backend build uses.
 */
void embed_spirv_modules(const Context &context, const Options &opt, std::span<const std::filesystem::path> modules) {
    LUISA_ASSERT(!modules.empty(), "embed received no SPIR-V modules.");
    auto embedder = opt.embedder.empty() ? find_host_tool(context, "luisa-embed-device-lib") : opt.embedder;
    LUISA_ASSERT(!embedder.empty(),
                 "Embedding needs the luisa-embed-device-lib host tool: build it with "
                 "`xmake build lc-vk-embed-device-lib` or pass --embedder <exe>.");
      auto arguments = std::vector<std::string>{};
      for (auto &&module : modules) {
          arguments.emplace_back(module.string());
      }
      for (auto &&argument : {"--unsigned"s, "--preserve-ext"s, "--prefix"s, "luisa_compute_vk_builtin_"s,
                              "-o"s, opt.embedded_source.string()}) {
          arguments.emplace_back(std::move(argument));
      }
    if (!opt.embedded_header.empty()) {
        arguments.emplace_back("-h");
        arguments.emplace_back(opt.embedded_header.string());
    }
    auto result = run_tool(embedder, arguments);
    if (!result.ok()) {
        LUISA_ERROR("luisa-embed-device-lib failed (exit {}):\n{}", result.exit_code, result.output);
    }
    if (!std::filesystem::is_regular_file(opt.embedded_source)) {
        LUISA_ERROR("luisa-embed-device-lib did not write {}.", opt.embedded_source.string());
    }
    LUISA_INFO("Embedded {} SPIR-V module(s) -> {}{}",
               modules.size(), opt.embedded_source.string(),
               opt.embedded_header.empty() ? "" : " + " + opt.embedded_header.string());
}
/**
 * \brief Read a stored artifact back through the runtime \c BinaryIO interface,
 * exactly the way `read_binary_io(SerdeType::kBuiltin, ...)` does.
 */
[[nodiscard]] std::vector<std::byte> read_artifact_bytes(const std::filesystem::path &path) {
    auto io = DirectoryBinaryIO{path.parent_path(), path.parent_path()};
    auto name = std::string{path.filename().string()};
    auto stream = io.read_internal_shader(name);
    if (!stream) {
        LUISA_ERROR("Artifact '{}' could not be opened through BinaryIO::read_internal_shader.", path.string());
    }
    auto bytes = std::vector<std::byte>(stream->length());
    if (!bytes.empty()) {
        stream->read({bytes.data(), bytes.size()});
    }
    return bytes;
}
/**
 * \brief Decode a DX `ShaderSerializer` v5 container read back through the
 * runtime `BinaryIO` interface and validate its contract.
 */
[[nodiscard]] bool verify_dx_artifact(const std::filesystem::path &path, std::string_view name) {
#ifdef _WIN32
    auto bytes = read_artifact_bytes(path);
    auto total = bytes.size();
    if (total <= sizeof(DxShaderSerHeader)) {
        LUISA_ERROR("DX artifact '{}' is truncated: {} bytes <= {}.", name, total, sizeof(DxShaderSerHeader));
    }
    auto header = DxShaderSerHeader{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.header_version != kDxHeaderVersion) {
        LUISA_ERROR("DX artifact '{}' has header version {} (expected {}).", name, header.header_version, kDxHeaderVersion);
    }
    if (header.root_sig_bytes == 0u || header.code_bytes == 0u) {
        LUISA_ERROR("DX artifact '{}' has an empty root signature or code blob.", name);
    }
    if (header.printer_count != 0u || header.bindless_count != 0u) {
        LUISA_ERROR("DX artifact '{}' declares {} printers / {} bindless heaps; builtins carry neither.",
                    name, header.printer_count, header.bindless_count);
    }
    // `dx::ShaderSerializer::Serialize` writes no kernel arguments for a builtin
    // (the codegen path has no AST Function), and `DeSerialize` reads them as one
    // blob after the code.
    if (header.kernel_arg_count != 0u) {
        LUISA_ERROR("DX artifact '{}' declares {} kernel arguments; builtins carry none.", name, header.kernel_arg_count);
    }
    for (auto size : header.block_size) {
        if (size == 0u) {
            LUISA_ERROR("DX artifact '{}' declares a zero thread-block dimension.", name);
        }
    }
    // Layout: [header][printers][root signature][code][properties][kernel args]
    auto expected = sizeof(DxShaderSerHeader) + header.root_sig_bytes + header.code_bytes +
                    header.property_count * sizeof(hlsl::Property);
    if (expected != total) {
        LUISA_ERROR("DX artifact '{}' has inconsistent section sizes: container {} bytes, header implies {}.",
                    name, total, expected);
    }
    auto *root_sig = reinterpret_cast<char const *>(bytes.data() + sizeof(header));
    auto *code = reinterpret_cast<char const *>(bytes.data() + sizeof(header) + header.root_sig_bytes);
    // The code blob must be a DXBC container so that
    // CreateComputePipelineState (psoDesc.CS.pShaderBytecode) can consume it.
    if (std::string_view{code, 4u} != "DXBC") {
        LUISA_ERROR("DX artifact '{}' code blob is not a DXBC container ('{}').", name, std::string_view{code, 4u});
    }
    auto properties = std::span<const hlsl::Property>{
        reinterpret_cast<hlsl::Property const *>(code + header.code_bytes),
        header.property_count};
    LUISA_INFO("DX artifact '{}' is loadable: header v{}, md5 {}, type md5 {}, block "
               "({}, {}, {}), {} properties, root signature {} bytes ('{}'), DXBC {} bytes, "
               "container {} bytes.",
               name, header.header_version, header.md5.to_string(false), header.type_md5.to_string(false),
               header.block_size[0], header.block_size[1], header.block_size[2],
               properties.size(), header.root_sig_bytes, std::string_view{root_sig, 4u},
               header.code_bytes, total);
    for (auto &&property : properties) {
        LUISA_INFO("  property: type={} space={} register={} array_size={}",
                   static_cast<int>(property.type), property.space_index,
                   property.register_index, property.array_size);
    }
    return true;
#else
    return false;
#endif
}
/**
 * \brief Decode a VK `ShaderSerializer` v10 container read back through the
 * runtime `BinaryIO` interface and validate the contract the backend loader
 * checks (version, semantic header, digests, sizes, SPIR-V header, interface).
 */
[[nodiscard]] bool verify_vk_artifact(const std::filesystem::path &path, std::string_view name) {
    auto bytes = read_artifact_bytes(path);
    auto total = bytes.size();
    if (total < sizeof(VkShaderSerHeader)) {
        LUISA_ERROR("VK artifact '{}' is truncated: {} bytes < {}.", name, total, sizeof(VkShaderSerHeader));
    }
    auto header = VkShaderSerHeader{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.header_ver != kVkShaderSerVersion) {
        LUISA_ERROR("VK artifact '{}' has header version {} (expected {}).", name, header.header_ver, kVkShaderSerVersion);
    }
    if (header.pipeline_ver != kVkXirPipelineVersion) {
        LUISA_ERROR("VK artifact '{}' has pipeline version {} (expected {}).", name, header.pipeline_ver, kVkXirPipelineVersion);
    }
    if (vk_semantic_header_md5(header) != header.semantic_header_md5) {
        LUISA_ERROR("VK artifact '{}' has a corrupt semantic header (digest mismatch).", name);
    }
    if (header.codegen_dialect != kVkDialectHlslSpirv && header.codegen_dialect != kVkDialectVulkanBuiltin) {
        LUISA_ERROR("VK artifact '{}' declares codegen dialect {}, which the builtin loaders do not accept.",
                    name, static_cast<uint32_t>(header.codegen_dialect));
    }
    if (header.use_bindless_buffer > 1u || header.use_bindless_tex2d > 1u || header.use_bindless_tex3d > 1u) {
        LUISA_ERROR("VK artifact '{}' has a non-canonical bindless flag.", name);
    }
    if (header.required_spirv_features != 0u) {
        LUISA_ERROR("VK artifact '{}' claims SPIR-V features 0x{:016x}; builtins must not.",
                    name, header.required_spirv_features);
    }
    for (auto size : header.block_size) {
        if (size == 0u) {
            LUISA_ERROR("VK artifact '{}' declares a zero thread-block dimension.", name);
        }
    }
    uint64_t expected{sizeof(VkShaderSerHeader)};
    expected += header.property_size * sizeof(hlsl::Property);
    expected += header.spv_byte_size;
    expected += header.printer_size_bytes;
    expected += header.constant_ubo_size;
    if (header.kernel_arg_count != 0u) {
        LUISA_ERROR("VK artifact '{}' declares {} kernel arguments; builtins carry none.", name, header.kernel_arg_count);
    }
    if (expected != total) {
        LUISA_ERROR("VK artifact '{}' has inconsistent section sizes: container {} bytes, header implies {}.",
                    name, total, expected);
    }
    auto properties = std::vector<hlsl::Property>{};
    properties.resize(static_cast<size_t>(header.property_size));
    auto property_bytes = properties.size() * sizeof(hlsl::Property);
    if (property_bytes > 0u) {
        std::memcpy(properties.data(), bytes.data() + sizeof(header), property_bytes);
    }
    if (vk_property_md5(properties) != header.property_md5) {
        LUISA_ERROR("VK artifact '{}' property table digest mismatch.", name);
    }
    if (header.spv_byte_size < 5u * sizeof(uint32_t) || header.spv_byte_size % sizeof(uint32_t) != 0u) {
        LUISA_ERROR("VK artifact '{}' carries an invalid SPIR-V payload size ({}).", name, header.spv_byte_size);
    }
    auto *spirv = bytes.data() + sizeof(header) + property_bytes;
    if (md5_of(spirv, header.spv_byte_size) != header.spv_md5) {
        LUISA_ERROR("VK artifact '{}' SPIR-V payload digest mismatch.", name);
    }
    auto words = std::vector<uint32_t>(header.spv_byte_size / sizeof(uint32_t));
    std::memcpy(words.data(), spirv, header.spv_byte_size);
    // vk::detail::valid_spirv_header: magic, SPIR-V 1.0..1.6, nonzero id bound,
    // zero schema. `ComputeShader` feeds exactly these words to
    // vkCreateShaderModule.
    if (words[0] != kSpirvMagic || words[1] < 0x00010000u || words[1] > 0x00010600u ||
        words[3] == 0u || words[4] != 0u) {
        LUISA_ERROR("VK artifact '{}' SPIR-V header is invalid (magic 0x{:08x}, version 0x{:08x}, "
                    "id bound {}, schema {}).",
                    name, words[0], words[1], words[3], words[4]);
    }
    auto sampler_count = 0u;
    auto indirect_count = 0u;
    for (auto &&property : properties) {
        sampler_count += property.type == hlsl::ShaderVariableType::SamplerHeap ? 1u : 0u;
        indirect_count += property.type == hlsl::ShaderVariableType::SPIRVIndirectDispatch ? 1u : 0u;
        if (property.type == hlsl::ShaderVariableType::ConstantValue &&
            (property.space_index != 0u || property.register_index != 0u || property.array_size != 1u)) {
            LUISA_ERROR("VK artifact '{}' declares a non-canonical push-constant property.", name);
        }
        if (property.type == hlsl::ShaderVariableType::SamplerHeap &&
            (property.space_index != 1u || property.register_index != 0u || property.array_size != kVulkanSamplerCount)) {
            LUISA_ERROR("VK artifact '{}' declares a non-canonical sampler heap (must be space 1, "
                        "register 0, {} descriptors).", name, kVulkanSamplerCount);
        }
    }
    if (indirect_count > 1u) {
        LUISA_ERROR("VK artifact '{}' declares {} indirect-dispatch properties (at most one).", name, indirect_count);
    }
    // vk::detail::plan_shader_interface requires exactly one canonical sampler
    // heap for every Vulkan shader interface (MISSING_OR_DUPLICATE_SAMPLER),
    // including the builtins loaded from `vulkan_builtin_buffer_properties`.
    if (sampler_count != 1u) {
        LUISA_ERROR("VK artifact '{}' declares {} sampler heaps; the descriptor interface requires "
                    "exactly one (space 1, register 0, {} descriptors). Pass --contract or --sampler-heap.",
                    name, sampler_count, kVulkanSamplerCount);
    }
    LUISA_INFO("VK artifact '{}' is loadable: dialect {}, header v{} pipeline v{}, md5 {}, "
               "spv md5 {}, block ({}, {}, {}), {} properties, SPIR-V {} words (version 0x{:08x}), "
               "container {} bytes.",
               name, static_cast<uint32_t>(header.codegen_dialect), header.header_ver, header.pipeline_ver,
               header.md5.to_string(false), header.spv_md5.to_string(false),
               header.block_size[0], header.block_size[1], header.block_size[2],
               header.property_size, words.size(), words[1], total);
    return true;
}
/**
 * \brief Validate an artifact of the requested dialect.
 */
[[nodiscard]] bool verify_artifact(Target target, const std::filesystem::path &path, std::string_view name) {
    return target == Target::DX ? verify_dx_artifact(path, name) : verify_vk_artifact(path, name);
}
/**
 * \brief Peek at a container's version fields before validating it.
 *
 * `inspect` runs on files this tool did not produce - including the superseded
 * v2 Vulkan builtin blobs - and a stale container is a finding to report, not a
 * fatal error. Returns nullopt when the file is too small to carry a version.
 */
[[nodiscard]] luisa::optional<bool> artifact_version_supported(Target target, const std::filesystem::path &path) {
    auto bytes = read_bytes(path);
    if (target == Target::DX) {
        if (bytes.size() < sizeof(uint64_t)) { return luisa::nullopt; }
        auto version = uint64_t{};
        std::memcpy(&version, bytes.data(), sizeof(version));
        if (version != kDxHeaderVersion) {
            LUISA_WARNING("'{}' is a DX ShaderSerializer v{} container; this tool reads and writes v{}."
                            " (the legacy *_vk.dxil blobs predate it).", path.string(), version, kDxHeaderVersion);
            return false;
        }
        return true;
    }
    if (bytes.size() < sizeof(uint64_t) + sizeof(uint32_t)) { return luisa::nullopt; }
    auto header_ver = uint64_t{};
    std::memcpy(&header_ver, bytes.data(), sizeof(header_ver));
    auto pipeline_ver = uint32_t{};
    std::memcpy(&pipeline_ver, bytes.data() + sizeof(uint64_t), sizeof(pipeline_ver));
    if (header_ver != kVkShaderSerVersion) {
        LUISA_WARNING("'{}' is a VK ShaderSerializer v{} container; this tool reads and writes v{}."
                        " (the superseded *_vk.dxil builtins use v2).", path.string(), header_ver, kVkShaderSerVersion);
        return false;
    }
    if (pipeline_ver != kVkXirPipelineVersion) {
        LUISA_WARNING("'{}' declares XIR pipeline version {} (expected {}).", path.string(), pipeline_ver, kVkXirPipelineVersion);
        return false;
    }
    return true;
}
/**
 * \brief Decode the literal string operand of a SPIR-V instruction, starting at
 * `index` (its first word) and stopping at the first NUL byte.
 */
[[nodiscard]] std::string spirv_string_operand(std::span<const uint32_t> words, size_t index, size_t limit) {
    auto text = std::string{};
    for (auto i = index; i < limit && i < words.size(); ++i) {
        auto word = words[i];
        for (size_t b = 0u; b < 4u; ++b) {
            auto c = static_cast<char>((word >> (8u * b)) & 0xffu);
            if (c == '\0') { return text; }
            text.push_back(c);
        }
    }
    return text;
}
/**
 * \brief The parts of a SPIR-V module the runtime loader and the builtin
 * contract depend on: the entry-point name (`spirv_has_entry_point` looks for
 * "main") and the LocalSize execution mode (the thread-block size).
 */
struct SpirvModuleInfo {
    std::string entry_name;
    uint32_t block_size[3]{0u, 0u, 0u};
    bool has_local_size{false};
    uint32_t version{0u};
    uint32_t id_bound{0u};
    [[nodiscard]] bool valid_header() const noexcept {
        return version >= 0x00010000u && version <= 0x00010600u && id_bound != 0u;
    }
};
[[nodiscard]] SpirvModuleInfo inspect_spirv_module(std::span<const uint32_t> words) {
    auto info = SpirvModuleInfo{};
    if (words.size() < 5u) { return info; }
    info.version = words[1];
    info.id_bound = words[3];
    for (size_t i = 5u; i < words.size();) {
        auto wc = static_cast<size_t>(words[i] >> 16u);
        auto opcode = words[i] & 0xffffu;
        if (wc == 0u || i + wc > words.size()) { break; }
        // OpEntryPoint: <execution> <entry-point-id> <name> <interface...>
        if (opcode == 15u && wc >= 4u) {
            info.entry_name = spirv_string_operand(words, i + 3u, i + wc);
        } else if (opcode == 16u && wc >= 3u && words[i + 2u] == 17u) {// OpExecutionMode LocalSize
            info.block_size[0] = wc > 3u ? words[i + 3u] : 1u;
            info.block_size[1] = wc > 4u ? words[i + 4u] : 1u;
            info.block_size[2] = wc > 5u ? words[i + 5u] : 1u;
            info.has_local_size = true;
        }
        i += wc;
    }
    return info;
}
/**
 * \brief Interpret a freshly compiled SPIR-V module. Its `main` entry point and
 * `LocalSize` execution mode are authoritative, so the block size recorded in a
 * container always matches what the driver will dispatch with.
 */
[[nodiscard]] SpirvModuleInfo describe_spirv(std::span<const std::byte> module, std::string_view name) {
    LUISA_ASSERT(module.size() >= 5u * sizeof(uint32_t) && module.size() % sizeof(uint32_t) == 0u,
                 "SPIR-V module '{}' is not a word-aligned module ({} bytes).", name, module.size());
    auto words = std::vector<uint32_t>(module.size() / sizeof(uint32_t));
    std::memcpy(words.data(), module.data(), module.size());
    auto magic = uint32_t{};
    std::memcpy(&magic, module.data(), sizeof(magic));
    LUISA_ASSERT(magic == kSpirvMagic, "SPIR-V module '{}' has magic 0x{:08x} (expected 0x07230203).",
                 name, magic);
    auto info = inspect_spirv_module(words);
    LUISA_ASSERT(info.valid_header() && words[4] == 0u,
                 "SPIR-V module '{}' has an invalid header (version 0x{:08x}, id bound {}, schema {}).",
                 name, info.version, info.id_bound, words[4]);
    LUISA_ASSERT(info.entry_name == "main",
                 "SPIR-V module '{}' declares the entry point '{}', but the Vulkan loader and the "
                 "codec (spirv_has_entry_point) require 'main'. Pass --entry to match the source.",
                 name, info.entry_name);
    return info;
}
/**
 * \brief Validate a bare SPIR-V module (the `spv` route's default output).
 */
[[nodiscard]] bool verify_spirv_module(const std::filesystem::path &path, std::string_view name) {
    auto bytes = read_bytes(path);
    auto info = describe_spirv(bytes, name);
    LUISA_INFO("SPIR-V module '{}': {} words ({} bytes), version 0x{:08x}, id bound {}, "
               "entry point '{}', block ({}, {}, {}){}",
               name, bytes.size() / sizeof(uint32_t), bytes.size(), info.version, info.id_bound,
               info.entry_name, info.block_size[0], info.block_size[1], info.block_size[2],
               info.has_local_size ? "" : " [no LocalSize execution mode]");
    return true;
}

/**
 * \brief Copy the artifact into the runtime's own data/cache store so a backend
 * device can pick it up without a rebuild (`--install`).
 */
void install_artifact(const Context &context, const Options &opt, std::span<const std::byte> artifact) {
    auto dir = opt.store_cache
                   ? context.create_runtime_subdir(".cache")
                   : context.create_runtime_subdir(".data");
    auto io = DirectoryBinaryIO{dir, dir};
    auto name = std::filesystem::path{opt.output}.filename().string();
    auto path = opt.store_cache
                    ? io.write_shader_cache(name, artifact)
                    : io.write_internal_shader(name, artifact);
    LUISA_INFO("Installed builtin '{}' into {} ({}) - readable through BinaryIO::{}.",
               opt.name, opt.store_cache ? "the shader cache" : "the builtin data store",
               path.string(), opt.store_cache ? "read_shader_cache" : "read_internal_shader");
}
}// namespace
int main(int argc, char *argv[]) {
    Options opt;
    if (!parse_options(argc, argv, opt)) {
        return EXIT_FAILURE;
    }
    const char *exe = (argc > 0 && argv[0]) ? argv[0] : "";
    Context context{exe};
    switch (opt.action) {
        case Action::INSPECT: {
            if (auto supported = artifact_version_supported(opt.target, opt.input); !supported || !*supported) {
                return EXIT_FAILURE;
            }
            if (verify_artifact(opt.target, opt.input, opt.name)) {
                return EXIT_SUCCESS;
            }
            return EXIT_FAILURE;
        }
        case Action::EMBED: {
            embed_spirv_modules(context, opt, opt.inputs);
            return EXIT_SUCCESS;
        }
        case Action::COMPILE:
            break;
    }
    auto source = read_file(opt.input);
    if (source.empty()) {
        LUISA_ERROR("Input shader file is empty: {}", opt.input.string());
    }
    if (!opt.prepend.empty()) {
        if (opt.target == Target::VK_BUILTIN) {
            LUISA_ERROR("--prepend applies to the dxc routes ('dx'/'vk') only; the "
                        "glslang route compiles the input file directly.");
        }
        auto header = read_file(opt.prepend);
        if (header.empty()) {
            LUISA_ERROR("Prepend file is empty: {}", opt.prepend.string());
        }
        source.insert(0, header);
    }
    auto source_view = std::string_view{source};
    auto contract_properties = std::vector<hlsl::Property>{};
    auto contract_block = std::array<uint32_t, 3>{};
    if (!opt.contract.empty()) {
        LUISA_ASSERT(opt.target == Target::VK_BUILTIN,
                     "--contract applies to the Vulkan builtin ('spv') target only.");
        LUISA_ASSERT(vk_builtin_contract(opt.contract, contract_properties, contract_block.data()),
                     "Unknown Vulkan builtin contract '{}': expected indirect_prepare, "
                     "accel_process or bindless_upload.",
                     opt.contract);
        // The mirrored layout contract is authoritative for the block size too.
        if (!opt.has_block_size) {
            std::memcpy(opt.block_size, contract_block.data(), sizeof(contract_block));
            opt.has_block_size = true;
        }
    }
    // The resource interface is derived from the source itself (or from the
    // builtin contract), so the emitted container matches the hand-written table
    // in `BuiltinKernel`.
    auto bindings = parse_bindings(source_view);
    if (bindings.empty() && contract_properties.empty()) {
        LUISA_ERROR("No :register(...) bindings found in {} - pass --contract for a "
                    "Vulkan builtin.", opt.input.string());
    }
    std::vector<hlsl::Property> properties;
    if (!contract_properties.empty()) {
        // The Vulkan builtin loader always attaches its own fixed table; use it
        // verbatim so the planned descriptor interface matches the runtime.
        properties = std::move(contract_properties);
    } else {
        properties.reserve(bindings.size() + 2u);
        for (auto &&b : bindings) {
            properties.emplace_back(hlsl::Property{b.type, b.space, b.register_index, b.array_size});
        }
        if (opt.push_constant_bytes >= 0) {
            // Vulkan: SPIRVIndirectDispatch carries the push-constant byte count
            // (descriptor_interface_plan.h). DX: ConstantValue carries it in
            // space_index as the number of 32-bit values.
            auto values = static_cast<uint32_t>((opt.push_constant_bytes + 3) / 4);
            if (opt.target == Target::VK_BUILTIN) {
                properties.emplace_back(hlsl::Property{hlsl::ShaderVariableType::SPIRVIndirectDispatch, 0u, 0u, values});
            } else {
                properties.emplace_back(hlsl::Property{hlsl::ShaderVariableType::ConstantValue, values, 0u, 1u});
            }
        }
        if (opt.sampler_heap) {
            properties.emplace_back(hlsl::Property{hlsl::ShaderVariableType::SamplerHeap, 1u, 0u, kVulkanSamplerCount});
        }
    }
    // Compile with the very same compiler path the backends (and the Vulkan
    // backend build) use.
    auto bytecode = std::vector<std::byte>{};
    auto module_info = SpirvModuleInfo{};
    if (opt.target == Target::VK_BUILTIN) {
        bytecode = compile_with_glslang(context, opt);
        // The compiled module is authoritative for the thread-block size: the
        // `*.def` layout contracts keep `[numthreads(...)]` behind a macro
        // (LC_INDIRECT_PREPARE_BLOCK_SIZE, LC_VULKAN_ACCEL_BLOCK_SIZE), and the
        // container must carry the same value `LocalSize` declares, because that
        // is what the Vulkan loader dispatches with.
        module_info = describe_spirv(bytecode, opt.name);
        if (opt.has_block_size) {
            LUISA_ASSERT(module_info.block_size[0] == opt.block_size[0] &&
                             module_info.block_size[1] == opt.block_size[1] &&
                             module_info.block_size[2] == opt.block_size[2],
                         "The requested block size ({}, {}, {}) does not match the SPIR-V "
                         "LocalSize ({}, {}, {}) of '{}'.",
                         opt.block_size[0], opt.block_size[1], opt.block_size[2],
                         module_info.block_size[0], module_info.block_size[1],
                         module_info.block_size[2], opt.input.string());
        } else {
            std::memcpy(opt.block_size, module_info.block_size, sizeof(opt.block_size));
            opt.has_block_size = true;
        }
    } else {
        if (!opt.has_block_size) {
            uint32_t block[3]{1u, 1u, 1u};
            auto constants = parse_scalar_consts(source_view);
            if (parse_block_size(source_view, constants, block)) {
                std::memcpy(opt.block_size, block, sizeof(block));
            } else {
                LUISA_WARNING(
                    "Cannot read a literal [numthreads(...)] from {} - using block size ({}, {}, {}). "
                    "Pass --block-size or --contract to override.",
                    opt.input.string(), opt.block_size[0], opt.block_size[1], opt.block_size[2]);
            }
        }
        if (opt.shader_model < 10u) {
            LUISA_ERROR("Illegal shader model: {}", opt.shader_model);
        }
        hlsl::ShaderCompiler compiler{context.runtime_directory(), opt.is_spirv()};
        auto dxc_args = dxc_arguments(opt);
        auto wide_args = std::vector<LPCWSTR>{};
        wide_args.reserve(dxc_args.size());
        for (auto &&a : dxc_args) { wide_args.emplace_back(a.c_str()); }
        auto code = compiler.compile(source_view, wide_args);
        bytecode = code.multi_visit_or(
            std::vector<std::byte>{},
            [&](hlsl::ComUniquePtr<IDxcBlob> &blob) {
                auto *begin = reinterpret_cast<std::byte const *>(blob->GetBufferPointer());
                return std::vector<std::byte>{begin, begin + blob->GetBufferSize()};
            },
            [&](auto &&error) {
                LUISA_ERROR("DXC compile error for builtin '{}': {}", opt.name, std::string_view{error});
                return std::vector<std::byte>{};
            });
    }
    if (bytecode.empty()) {
        LUISA_ERROR("The compiler produced no bytecode for builtin '{}'.", opt.name);
    }
    if (opt.target == Target::VK_BUILTIN && !module_info.has_local_size) {
        LUISA_WARNING("SPIR-V module for builtin '{}' declares no LocalSize execution mode; "
                      "the thread-block size ({}, {}, {}) is taken from the source.",
                      opt.name, opt.block_size[0], opt.block_size[1], opt.block_size[2]);
    }
    auto package = !opt.raw && (opt.is_vk_container() || opt.target == Target::DX);
    auto artifact = !package
                        ? std::move(bytecode)
                        : (opt.target == Target::DX
                               ? encode_dx_artifact(opt, source_view, properties, bytecode)
                               : encode_vk_artifact(opt, source_view, properties, bytecode));
    if (artifact.empty()) {
        LUISA_ERROR("Failed to encode the builtin artifact for '{}'.", opt.name);
    }
    write_bytes(opt.output, artifact);
    if (!std::filesystem::is_regular_file(opt.output)) {
        LUISA_ERROR("Bytecode was not written to {}", opt.output.string());
    }
    if (opt.verify) {
        auto ok = package
                      ? verify_artifact(opt.target, opt.output, opt.name)
                      : verify_spirv_module(opt.output, opt.name);
        if (!ok) {
            LUISA_ERROR("Artifact verification failed for '{}'.", opt.name);
        }
    }
    if (opt.install) {
        if (!package) {
            LUISA_WARNING("--install only stores container artifacts; {} written as a raw module is left alone.", opt.name);
        } else {
            install_artifact(context, opt, artifact);
        }
    }
    if (opt.embed) {
        // The embedded arrays hold raw SPIR-V, exactly like the build tree's
        // `vk_builtin/*.spv`, so reuse the freshly compiled module. The embedder
        // derives the symbol name from the file stem plus (with --preserve-ext)
        // the extension, so the module must be named `<kernel>.spv` to produce
        // `luisa_compute_vk_builtin_<kernel>_spv`.
        auto scratch_dir = std::filesystem::path{};
        auto module_path = std::filesystem::path{};
        if (opt.raw) {
            module_path = opt.output;// already the bare module
        } else {
            scratch_dir = opt.output.parent_path() / ".lc-compile-builtin";
            module_path = scratch_dir / (opt.name + ".spv");
            write_bytes(module_path, bytecode);
        }
        auto embed_options = opt;
        embed_options.embedded_source = opt.embedded_source.empty()
                                            ? opt.output.parent_path() / "vulkan_builtin_spirv_embedded.cpp"
                                            : opt.embedded_source;
        embed_options.embedded_header = opt.embedded_header.empty()
                                            ? opt.output.parent_path() / "vulkan_builtin_spirv_embedded.h"
                                            : opt.embedded_header;
        auto modules = std::vector<std::filesystem::path>{module_path};
        embed_spirv_modules(context, embed_options, modules);
        if (!scratch_dir.empty()) {
            std::error_code ec;
            std::filesystem::remove(module_path, ec);
            std::filesystem::remove(scratch_dir, ec);
        }
    }
    LUISA_INFO("Builtin '{}' ({}{}): {} properties, block ({}, {}, {}), {} bytes -> {}",
               opt.name,
               opt.target == Target::DX ? "dx" : (opt.target == Target::VK_BUILTIN ? "spv" : "vk"),
               package ? "" : ", raw",
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
