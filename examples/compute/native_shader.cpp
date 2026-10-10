// JSON-driven native shader dispatch (HLSL on dx/vk, GLSL on vk, CUDA C++ on
// cuda) through NativeShaderExt.
//
//   example_native_shader <backend> <dispatch.json> [shader...] [options]
//   example_native_shader <backend> <shader...>
//   example_native_shader <backend> --self-test
//
// This translation unit owns the command line, the mode orchestration, the
// shader registry glue, the interactive ImGui display pass and the self test.
// The dispatch document itself is parsed/serialized by
// native_shader_dispatch.cpp and executed by native_shader_runtime.cpp.
//
// The example never aborts on a reportable problem: every failure is printed to
// stderr and turns into a non-zero exit code. See
// native_shader_examples/README.md for the document schema.
#include <algorithm>
#include <cstdio>
#include <cstring>

#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/stream.h>

#ifdef LUISA_ENABLE_GUI
// `luisa/gui/imgui_window.h` only forward-declares its ImGui types; the GUI
// window's drawing code uses the Dear ImGui API directly (the include
// directory comes from the `lc-gui` / `luisa-compute-gui` target).
#include <imgui.h>
#include <luisa/gui/imgui_window.h>
#endif

#include "native_shader_dispatch.h"
#include "native_shader_embedded.h"
#include "native_shader_runtime.h"

using namespace luisa;
using namespace luisa::compute;
namespace ns = luisa::native_shader;

namespace {

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

// `LUISA_ERROR` is fatal (it aborts), which is wrong for a tool that must
// report several problems and exit with a code, so failures go to stderr.
// `--self-test`'s rejection corpus runs documents that are *supposed* to be
// refused, and collects their diagnostics through `RunReport`; printing them
// would look like a failure of the test itself.
bool g_quiet = false;

void report_failure(luisa::string_view message) noexcept {
    if (g_quiet) { return; }
    std::fprintf(stderr, "[native_shader] FAIL: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    std::fflush(stderr);
}

void report_warning(luisa::string_view message) noexcept {
    if (g_quiet) { return; }
    std::fprintf(stderr, "[native_shader] warning: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    std::fflush(stderr);
}

void report_diagnostics(const ns::Diagnostics &diagnostics) noexcept {
    for (auto &&error : diagnostics.errors) { report_failure(error); }
    for (auto &&warning : diagnostics.warnings) { report_warning(warning); }
}

// Creating a device for a backend the process does not have installed aborts
// inside the runtime, so the name is checked up front and reported with the
// list of what is available.
[[nodiscard]] bool backend_is_installed(const Context &context,
                                        luisa::string_view backend) noexcept {
    auto available = luisa::string{};
    for (auto &&name : context.installed_backends()) {
        if (name == backend) { return true; }
        if (!available.empty()) { available.append(", "); }
        available.append(name);
    }
    report_failure(luisa::format("backend '{}' is not installed (available: {})",
                                 backend, available));
    return false;
}

void apply_log_level(luisa::string_view level) noexcept {
    if (level == "verbose") {
        log_level_verbose();
    } else if (level == "warning") {
        log_level_warning();
    } else if (level == "error") {
        log_level_error();
    } else {
        log_level_info();
    }
}

// ---------------------------------------------------------------------------
// command line
// ---------------------------------------------------------------------------

struct Options {
    luisa::string backend;
    luisa::string document;// empty: use the embedded default document
    luisa::vector<luisa::string> shader_paths;
    bool has_mode_override{false};
    bool interactive{false};
    bool has_frames{false};
    uint32_t frames{1u};
    bool has_exit_after_frames{false};
    uint32_t exit_after_frames{0u};
    bool has_output_dir{false};
    luisa::string output_dir;
    bool has_workdir{false};
    luisa::string workdir;
    bool strict{false};
    bool no_gui{false};
    bool stats{false};
    bool dump_dispatch{false};
    luisa::string dump_path;
    bool self_test{false};
    bool sync_uploads{false};
    bool has_entry_point{false};
    luisa::string entry_point;
    bool has_push_constant_size{false};
    uint32_t push_constant_size{0u};
    bool has_language{false};
    luisa::string language;
};

void print_usage(luisa::string_view program) noexcept {
    std::printf(
        "usage: %.*s <backend> [dispatch.json] [shader...] [options]\n"
        "\n"
        "  <backend>        dx | vk | cuda | ... (any backend the runtime finds)\n"
        "  dispatch.json    a dispatch document; without one the embedded default\n"
        "                   workflow (upload -> native dispatch -> verified readback)\n"
        "                   runs with the shaders given on the command line\n"
        "  shader...        native shader sources; they are appended to the document's\n"
        "                   `shaders` and override an entry with the same name\n"
        "\n"
        "options:\n"
        "  --offline              run the workflow without a window (default)\n"
        "  --interactive          open the ImGui window and display `mode.display_image`\n"
        "  --frames N             offline: run the workflow N times\n"
        "  --exit-after-frames N  interactive: close the window after N frames\n"
        "  --output-dir DIR       prefix for relative output sinks\n"
        "  --workdir DIR          resolve the document's relative paths against DIR\n"
        "  --strict               turn warnings (unknown keys, overrides) into errors\n"
        "  --no-gui               interactive is then an error (never a silent fallback)\n"
        "  --stats                print per-stage timings\n"
        "  --sync-uploads         synchronize after each upload instead of using events\n"
        "  --dump-dispatch FILE   write the effective document to FILE and exit\n"
        "  --self-test            codec round-trip, corpus cross-check, negative and\n"
        "                         execution corpora; exit code 0 iff everything passed\n"
        "  --print-schema         print a machine-readable schema summary and exit\n"
        "  --entry NAME           entry point for the shaders given on the command line\n"
        "  --push-constant-size N uniform bytes for the command-line shaders\n"
        "  --language LANG        hlsl | glsl | cuda_nvrtc for the command-line shaders\n"
        "  --help                 this text\n"
        "\n"
        "Paths inside the document resolve against the document's directory (or\n"
        "--workdir); relative output sinks additionally get --output-dir as a prefix.\n"
        "The process exits non-zero on the first failure.\n",
        static_cast<int>(program.size()), program.data());
}

void print_schema() noexcept {
    // Every list here comes from the codec's own tables, so the printed schema and
    // the validator cannot disagree.
    auto print_list = [](std::string_view key,
                         luisa::span<const luisa::string_view> names,
                         bool trailing_comma) noexcept {
        std::printf("  \"%.*s\": [", static_cast<int>(key.size()), key.data());
        for (auto i = size_t{0u}; i < names.size(); i++) {
            std::printf("%s\"%.*s\"", i == 0u ? "" : ", ",
                        static_cast<int>(names[i].size()), names[i].data());
        }
        std::printf("]%s\n", trailing_comma ? "," : "");
    };
    std::printf("{\n"
                "  \"schema\": \"luisa.native_shader.dispatch\",\n"
                "  \"version\": 1,\n");
    print_list("command_kinds", ns::command_kind_spellings(), true);
    print_list("resource_types", ns::resource_type_spellings(), true);
    print_list("unsupported_command_kinds", ns::unsupported_command_kinds(), true);
    print_list("unsupported_resource_types", ns::unsupported_resource_types(), true);
    print_list("buffer_elements", ns::buffer_element_spellings(), true);
    print_list("pixel_storages", ns::pixel_storage_spellings(), true);
    print_list("usages", ns::usage_spellings(), true);
    print_list("dsl_kernels", ns::builtin_dsl_kernels(), true);
    std::printf("  \"custom_commands\": {\"native_shader_dispatch\": 1536, \"dstorage_read\": 512},\n"
                "  \"limits\": {");
    // The budgets come from the codec's default `JsonLimits`, so the printed
    // schema and the validator cannot disagree either.
    auto limits = ns::JsonLimits{};
    std::printf("\"max_document_bytes\": %zu, \"max_resources\": %zu, "
                "\"max_shaders\": %zu, \"max_commands\": %zu, "
                "\"max_inline_bytes\": %zu, \"max_uniform_bytes\": %zu, "
                "\"max_bindings_per_dispatch\": %zu, \"max_resource_bytes\": %zu, "
                "\"max_string_bytes\": %zu, \"max_errors\": %zu, \"max_depth\": %u},\n",
                limits.max_document_bytes, limits.max_resources, limits.max_shaders,
                limits.max_commands, limits.max_inline_bytes, limits.max_uniform_bytes,
                limits.max_bindings_per_dispatch, limits.max_resource_bytes,
                limits.max_string_bytes, limits.max_errors, limits.max_depth);
    std::printf("  \"samples\": [\"scale_offline.json\", \"scale_interactive.json\",\n"
                "    \"all_commands_offline.json\"],\n"
                "  \"exit_codes\": {\"0\": \"success\", \"non-zero\": \"the first reported failure\"}\n"
                "}\n");
}

// Parses a non-negative decimal integer without touching <cstdlib>'s parsing
// (which silently accepts trailing garbage).
[[nodiscard]] bool parse_uint32(luisa::string_view text, uint32_t &out) noexcept {
    if (text.empty()) { return false; }
    auto value = uint64_t{0u};
    for (auto c : text) {
        if (c < '0' || c > '9') { return false; }
        value = value * 10u + static_cast<uint64_t>(c - '0');
        if (value > 0xffffffffull) { return false; }
    }
    out = static_cast<uint32_t>(value);
    return true;
}

// `arguments` is the command line without the program name: the backend first,
// then a dispatch document (optional), shaders and options in any order.
[[nodiscard]] bool parse_cli(luisa::span<const luisa::string_view> arguments,
                             Options &options, luisa::string &error) noexcept {
    if (arguments.empty()) { return false; }
    options.backend = arguments.front();
    for (auto i = size_t{1u}; i < arguments.size(); i++) {
        auto arg = arguments[i];
        auto value_of = [&](luisa::string_view name) noexcept -> luisa::optional<luisa::string_view> {
            if (i + 1u >= arguments.size()) {
                error = luisa::format("missing value for {}", name);
                return luisa::nullopt;
            }
            return arguments[++i];
        };
        auto uint_of = [&](luisa::string_view name, uint32_t &out) noexcept {
            auto value = value_of(name);
            if (!value.has_value()) { return false; }
            if (!parse_uint32(*value, out)) {
                error = luisa::format("'{}' needs a non-negative integer, got '{}'", name, *value);
                return false;
            }
            return true;
        };
        if (arg == "--offline") {
            options.has_mode_override = true;
            options.interactive = false;
        } else if (arg == "--interactive") {
            options.has_mode_override = true;
            options.interactive = true;
        } else if (arg == "--frames") {
            if (!uint_of(arg, options.frames)) { return false; }
            options.has_frames = true;
        } else if (arg == "--exit-after-frames") {
            if (!uint_of(arg, options.exit_after_frames)) { return false; }
            options.has_exit_after_frames = true;
        } else if (arg == "--push-constant-size") {
            if (!uint_of(arg, options.push_constant_size)) { return false; }
            options.has_push_constant_size = true;
        } else if (arg == "--output-dir") {
            auto value = value_of(arg);
            if (!value.has_value()) { return false; }
            options.has_output_dir = true;
            options.output_dir = luisa::string{value->data(), value->size()};
        } else if (arg == "--workdir") {
            auto value = value_of(arg);
            if (!value.has_value()) { return false; }
            options.has_workdir = true;
            options.workdir = luisa::string{value->data(), value->size()};
        } else if (arg == "--dump-dispatch") {
            auto value = value_of(arg);
            if (!value.has_value()) { return false; }
            options.dump_dispatch = true;
            options.dump_path = luisa::string{value->data(), value->size()};
        } else if (arg == "--entry") {
            auto value = value_of(arg);
            if (!value.has_value()) { return false; }
            options.has_entry_point = true;
            options.entry_point = luisa::string{value->data(), value->size()};
        } else if (arg == "--language") {
            auto value = value_of(arg);
            if (!value.has_value()) { return false; }
            options.has_language = true;
            options.language = luisa::string{value->data(), value->size()};
        } else if (arg == "--strict") {
            options.strict = true;
        } else if (arg == "--no-gui") {
            options.no_gui = true;
        } else if (arg == "--stats") {
            options.stats = true;
        } else if (arg == "--sync-uploads") {
            options.sync_uploads = true;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg.starts_with("--")) {
            error = luisa::format("unknown option '{}'", arg);
            return false;
        } else if (arg.ends_with(".json")) {
            if (!options.document.empty()) {
                error = luisa::format("more than one dispatch document ('{}' and '{}')",
                                      options.document, arg);
                return false;
            }
            options.document = luisa::string{arg.data(), arg.size()};
        } else {
            options.shader_paths.emplace_back(arg.data(), arg.size());
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// documents
// ---------------------------------------------------------------------------

// The file name stem with everything outside [A-Za-z0-9_.-] replaced by '_',
// which is the name a command-line shader gets when the document does not
// mention it.
[[nodiscard]] luisa::string sanitized_stem(luisa::string_view path) noexcept {
    auto file_path = luisa::filesystem::path{};
    if (!luisa::path_from_narrow(path, file_path)) { return luisa::string{"shader"}; }
    auto file_name = file_path.filename().string();
    auto dot = file_name.find_last_of('.');
    auto stem = dot == std::string::npos ? file_name : file_name.substr(0u, dot);
    luisa::string result;
    result.reserve(stem.size());
    for (auto c : stem) {
        auto ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        result.push_back(ok ? c : '_');
    }
    if (result.empty()) { result = "shader"; }
    return result;
}

[[nodiscard]] ns::ShaderJson cli_shader_entry(const Options &options,
                                              luisa::string_view path) noexcept {
    ns::ShaderJson shader;
    shader.name = sanitized_stem(path);
    shader.source_is_file = true;
    shader.path = path;
    shader.entry_point = "main";
    if (options.has_entry_point) { shader.entry_point = options.entry_point; }
    if (options.has_push_constant_size) {
        shader.push_constant_size = options.push_constant_size;
    }
    if (options.has_language) {
        auto language = NativeShaderLanguage::HLSL;
        if (ns::parse_native_shader_language(options.language, language)) {
            shader.language = language;
            shader.has_language = true;
        }
    }
    shader.from_cli = true;
    return shader;
}

// The document used when the command line carries no JSON: the embedded default
// workflow, plus either the embedded shader for the selected backend (no
// command-line shader) or the command-line shaders (the workflow's `scale`
// shader is the single command-line source when there is exactly one).
[[nodiscard]] bool build_default_document(luisa::string_view backend,
                                          const Options &options,
                                          ns::DispatchJson &document,
                                          luisa::string &error) noexcept {
    auto parsed = ns::parse_dispatch_json(ns::kEmbeddedDefaultDocument, ns::JsonLimits{});
    if (!parsed.value.has_value()) {
        error = luisa::format("the embedded default document does not parse: {}",
                              parsed.errors.empty() ? "no diagnostic" : parsed.errors.front());
        return false;
    }
    document = std::move(*parsed.value);
    if (options.shader_paths.empty()) {
        ns::ShaderJson shader;
        shader.name = "scale";
        shader.has_language = true;
        shader.push_constant_size = 2u * sizeof(float);
        if (backend == "cuda") {
            shader.language = NativeShaderLanguage::CUDA_NVRTC;
            shader.source = luisa::string{ns::kEmbeddedDefaultCudaSource.data(), ns::kEmbeddedDefaultCudaSource.size()};
            shader.entry_point = "scale";
            shader.block_size = uint3{64u, 1u, 1u};
        } else if (backend == "vk") {
            shader.language = NativeShaderLanguage::GLSL;
            shader.source = luisa::string{ns::kEmbeddedDefaultGlslSource.data(), ns::kEmbeddedDefaultGlslSource.size()};
            shader.entry_point = "main";
        } else {
            shader.language = NativeShaderLanguage::HLSL;
            shader.source = luisa::string{ns::kEmbeddedDefaultHlslSource.data(), ns::kEmbeddedDefaultHlslSource.size()};
            shader.entry_point = "CSMain";
        }
        document.shaders.emplace_back(std::move(shader));
        return true;
    }
    for (auto &&path : options.shader_paths) {
        auto shader = cli_shader_entry(options, path);
        // The embedded workflow binds the shader named `scale`; a single
        // command-line source therefore takes that name.
        if (options.shader_paths.size() == 1u) { shader.name = "scale"; }
        document.shaders.emplace_back(std::move(shader));
    }
    return true;
}

// Appends the command-line shaders to a document. A command-line shader
// replaces *every* document entry of the same name: a document may carry one
// variant per language, and an override has to win over all of them.
void merge_cli_shaders(const Options &options,
                       ns::DispatchJson &document,
                       ns::Diagnostics &diagnostics) noexcept {
    for (auto &&path : options.shader_paths) {
        auto shader = cli_shader_entry(options, path);
        auto replaced = false;
        for (auto iter = document.shaders.begin(); iter != document.shaders.end();) {
            if (iter->name == shader.name) {
                iter = document.shaders.erase(iter);
                replaced = true;
            } else {
                ++iter;
            }
        }
        if (replaced) {
            diagnostics.warning(luisa::format(
                "shaders: the command line overrides the shader named '{}'",
                shader.name));
        }
        document.shaders.emplace_back(std::move(shader));
    }
}

// Applies the CLI overrides that are mode/config level rather than shader level.
void apply_cli_overrides(const Options &options, ns::DispatchJson &document) noexcept {
    if (options.has_mode_override) { document.mode.interactive = options.interactive; }
    if (options.has_frames) { document.mode.frames = options.frames; }
    if (options.has_exit_after_frames) {
        document.mode.exit_after_frames = options.exit_after_frames;
    }
    if (options.no_gui) { document.mode.gui = false; }
    if (options.has_output_dir) { document.config.output_dir = options.output_dir; }
    if (options.strict) { document.config.strict = true; }
}

// The thread-group size of the two elementwise 2-D kernels. The display pass
// clamps to the image extent itself, so any window size is legal.
inline constexpr auto kDisplayBlockSize = uint3{16u, 8u, 1u};

// ---------------------------------------------------------------------------
// built-in DSL kernels
// ---------------------------------------------------------------------------

// Registers a compiled DSL kernel so that `shader_dispatch` can name it.
template<size_t N, typename... Args>
void register_dsl_kernel(ns::ShaderRegistry &registry,
                         luisa::string name,
                         Shader<N, Args...> shader) noexcept {
    auto handle = shader.handle();
    auto argument_count = decltype(shader)::arg_count();
    auto uniform_size = shader.uniform_size();
    auto block_size = shader.block_size();
    registry.add_dsl(std::move(name), ns::Owner::create(std::move(shader)),
                     handle, argument_count, uniform_size, block_size,
                     static_cast<uint32_t>(N));
}

void register_builtin_kernels(Device &device, ns::ShaderRegistry &registry) noexcept {
    // `x` is linear; the branch-free sRGB transfer is the one path_tracing.cpp
    // uses (`select` rather than a branch, which keeps the kernel uniform).
    Callable linear_to_srgb = [](Var<float3> x) noexcept {
        return saturate(select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                               12.92f * x,
                               x <= 0.00031308f));
    };
    // One thread per display pixel; `scale` is the HDR exposure. The extent
    // guard is required because the dispatch covers whole thread groups.
    Kernel2D hdr_to_display = [&](ImageFloat hdr, ImageFloat display,
                                  Float scale, Float width, Float height) noexcept {
        set_block_size(kDisplayBlockSize.x, kDisplayBlockSize.y, kDisplayBlockSize.z);
        UInt2 p = dispatch_id().xy();
        $if (cast<float>(p.x) < width) {
            $if (cast<float>(p.y) < height) {
                Float4 v = hdr.read(p) * scale;
                Float3 c = linear_to_srgb(clamp(v.xyz(), 0.0f, 1.0f));
                display.write(p, make_float4(c, 1.0f));
            };
        };
    };
    // Fills an HDR image with a gradient plus a checker, so that the display
    // pass has something with values above 1.0 to tone down. The pixel
    // coordinates are converted component-wise: the CUDA codegen has no
    // uint2 -> float2 `cast`.
    Kernel2D fill_hdr_gradient = [](ImageFloat image, Float width, Float height,
                                    Float exposure) noexcept {
        set_block_size(kDisplayBlockSize.x, kDisplayBlockSize.y, kDisplayBlockSize.z);
        UInt2 p = dispatch_id().xy();
        $if (cast<float>(p.x) < width) {
            $if (cast<float>(p.y) < height) {
                Float2 uv = make_float2((cast<float>(p.x) + 0.5f) / width,
                                        (cast<float>(p.y) + 0.5f) / height);
                Float checker = cast<float>((p.x / 32u + p.y / 32u) % 2u);
                Float3 hdr = make_float3(uv.x, uv.y, 0.25f) * exposure + checker * 0.5f;
                image.write(p, make_float4(hdr, 1.0f));
            };
        };
    };
    // The DSL twin of the native `scale` shader, used by the all-commands corpus.
    Kernel1D scale_buffer = [](BufferFloat src, BufferFloat dst,
                               Float k, Float c) noexcept {
        UInt i = dispatch_id().x;
        dst->write(i, src->read(i) * k + c);
    };
    register_dsl_kernel(registry, "hdr_to_display", device.compile(hdr_to_display));
    register_dsl_kernel(registry, "fill_hdr_gradient", device.compile(fill_hdr_gradient));
    register_dsl_kernel(registry, "scale_buffer", device.compile(scale_buffer));
}

// ---------------------------------------------------------------------------
// one document, end to end
// ---------------------------------------------------------------------------

[[nodiscard]] bool create_dstorage_stream(Device &device,
                                          const ns::DispatchJson &document,
                                          Stream &stream) noexcept {
    auto *dstorage = device.extension<DStorageExt>();
    if (dstorage == nullptr || !document.config.dstorage.enabled) { return false; }
    stream = dstorage->create_stream(DStorageStreamOption{
        .source = DStorageStreamSource::FileSource,
        .staging_buffer_size = document.config.dstorage.staging_buffer_size,
        .supports_hdd = false});
    return static_cast<bool>(stream);
}

#ifdef LUISA_ENABLE_GUI

// Builds the encoder for the display pass and enqueues it on `stream`: one
// thread per display pixel, with the kernel clamping to the image extent.
[[nodiscard]] bool dispatch_display(Stream &stream,
                                    const ns::ShaderRegistry &shaders,
                                    luisa::string_view kernel_name,
                                    const Image<float> &hdr,
                                    const Image<float> &destination,
                                    float scale,
                                    uint2 size,
                                    luisa::string &error) noexcept {
    constexpr auto kExpectedArgumentCount = 5u;
    auto *entry = shaders.find_dsl(kernel_name);
    if (entry == nullptr) {
        error = luisa::format("mode.display_kernel: no DSL kernel named '{}'", kernel_name);
        return false;
    }
    if (entry->argument_count != kExpectedArgumentCount) {
        error = luisa::format(
            "mode.display_kernel: '{}' takes {} argument(s), but the display pass "
            "supplies {} (ImageFloat hdr, ImageFloat display, Float scale, "
            "Float width, Float height)",
            kernel_name, entry->argument_count, kExpectedArgumentCount);
        return false;
    }
    auto width = static_cast<float>(size.x);
    auto height = static_cast<float>(size.y);
    ComputeDispatchCmdEncoder encoder{entry->handle, kExpectedArgumentCount,
                                      entry->uniform_size};
    encoder.encode_texture(hdr.handle(), 0u);
    encoder.encode_texture(destination.handle(), 0u);
    encoder.encode_uniform(&scale, sizeof(scale), alignof(float));
    encoder.encode_uniform(&width, sizeof(width), alignof(float));
    encoder.encode_uniform(&height, sizeof(height), alignof(float));
    encoder.set_dispatch_size(uint3{size.x, size.y, 1u});
    stream << std::move(encoder).build();
    return true;
}

// Interactive mode: run the workflow (once, or per frame), convert the HDR
// display image to the display destination and present it.
[[nodiscard]] int run_interactive(Device &device,
                                  Stream &stream,
                                  const Options &options,
                                  ns::DispatchJson &document,
                                  ns::ResourceRegistry &resources,
                                  ns::ShaderRegistry &shaders,
                                  ns::WorkflowExecutor &executor,
                                  const ns::PathResolver &paths) noexcept {
    auto *hdr_entry = resources.find(document.mode.display_image);
    if (hdr_entry == nullptr || hdr_entry->resource.float_image == nullptr) {
        report_failure(luisa::format(
            "mode.display_image: '{}' is not a float image resource",
            document.mode.display_image));
        return 1;
    }
    auto window_size = make_uint2(document.mode.window.width, document.mode.window.height);
    auto window = luisa::make_unique<ImGuiWindow>(
        device, stream, document.mode.window.title,
        ImGuiWindow::Config{.size = window_size, .vsync = document.mode.window.vsync});
    if (!window->valid()) {
        report_failure("the ImGui window could not be created");
        return 1;
    }
    // The destination is either the named texture resource or an internal image
    // sized like the window, which lives for as long as the loop below.
    Image<float> internal_destination;
    auto *destination = static_cast<const Image<float> *>(nullptr);
    if (document.mode.display_destination.empty() ||
        document.mode.display_destination == "auto") {
        internal_destination = device.create_image<float>(PixelStorage::FLOAT4, window_size, 1u);
        destination = &internal_destination;
    } else {
        auto *entry = resources.find(document.mode.display_destination);
        if (entry == nullptr || entry->resource.float_image == nullptr) {
            report_failure(luisa::format(
                "mode.display_destination: '{}' is not a float image resource",
                document.mode.display_destination));
            return 1;
        }
        destination = entry->resource.float_image;
    }
    auto texture = window->register_texture(*destination, Sampler::linear_linear_edge());
    if (document.mode.dispatch_per_frame && !executor.execute(document)) { return 1; }
    auto frame_index = 0u;
    auto snapshots = 0u;
    Clock clock;
    while (!window->should_close() &&
           (document.mode.exit_after_frames == 0u ||
            frame_index < document.mode.exit_after_frames)) {
        window->prepare_frame();
        if (auto error = luisa::string{}; !dispatch_display(
                stream, shaders, document.mode.display_kernel,
                *hdr_entry->resource.float_image, *destination,
                document.mode.display_scale, window_size, error)) {
            report_failure(error);
            return 1;
        }
        ImGui::SetNextWindowPos(ImVec2{0.0f, 0.0f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
        ImGui::Begin("native shader", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoSavedSettings);
        ImGui::Image(texture, ImGui::GetContentRegionAvail());
        ImGui::TextDisabled("'%s'  %ux%u  exposure %.3f  frame %u",
                            document.mode.display_image.c_str(),
                            window_size.x, window_size.y,
                            static_cast<double>(document.mode.display_scale),
                            frame_index);
        ImGui::End();
        window->render_frame();
        if (document.mode.snapshot.every != 0u &&
            frame_index % document.mode.snapshot.every == 0u) {
            auto path = paths.resolve_output(document.mode.snapshot.path);
            auto extent = make_uint3(window_size.x, window_size.y, 1u);
            luisa::vector<std::byte> pixels{pixel_storage_size(PixelStorage::FLOAT4, extent)};
            stream << destination->copy_to(luisa::span{pixels}) << synchronize();
            if (auto error = luisa::string{}; !ns::write_output(
                    path, luisa::span<const std::byte>{pixels.data(), pixels.size()},
                    "png", extent, PixelStorage::FLOAT4, true, error)) {
                report_failure(error);
                return 1;
            }
            snapshots++;
        }
        frame_index++;
    }
    window->destroy();
    LUISA_INFO("interactive session: {} frame(s) in {:.1f} ms ({} snapshot(s)).",
               frame_index, clock.toc(), snapshots);
    return 0;
}

#endif// LUISA_ENABLE_GUI

// `kind=count, ...` for the kinds a document actually ran: the machine-readable
// half of the end-of-run summary.
[[nodiscard]] luisa::string summarize_command_counts(
    luisa::span<const size_t> counts) noexcept {
    auto summary = luisa::string{};
    for (auto i = size_t{0u}; i < counts.size(); i++) {
        if (counts[i] == 0u) { continue; }
        if (!summary.empty()) { summary.append(", "); }
        summary.append(luisa::format("{}={}",
                                     ns::command_kind_name(
                                         static_cast<ns::CommandKind>(i)),
                                     counts[i]));
    }
    return summary.empty() ? luisa::string{"none"} : summary;
}

// What one `run_document` observed: the self test asserts on it, so the
// per-kind counts prove that every command of a corpus actually ran.
struct RunReport {
    ns::Diagnostics diagnostics;
    luisa::vector<size_t> command_counts;
};

// Runs one document to completion on `backend`. Used by `main` and by the self
// test, so it owns everything: device, streams, registries and sinks.
[[nodiscard]] int run_document(Context &context,
                               luisa::string_view backend,
                               const Options &options,
                               ns::DispatchJson document,
                               luisa::string_view document_dir,
                               RunReport *report = nullptr) noexcept {
    // The diagnostics and the per-kind counts outlive the body below, which
    // returns from many places: the one exit copies them out.
    auto diagnostics = ns::Diagnostics{};
    auto command_counts = luisa::vector<size_t>{};
    auto status = [&]() noexcept -> int {
        auto device = context.create_device(backend);
        if (!device) {
            report_failure(luisa::format("cannot create the '{}' device", backend));
            return 1;
        }
        apply_log_level(document.config.log_level);
        diagnostics.max_errors = document.config.limits.max_errors;
        ns::PathResolver paths;
        paths.set_diagnostics(diagnostics);
        paths.set_document_dir(document_dir);
        if (options.has_workdir) { paths.set_workdir(options.workdir); }
        paths.set_output_dir(document.config.output_dir);

        auto *native_ext = device.extension<NativeShaderExt>();
        if (native_ext == nullptr) {
            LUISA_INFO("backend '{}' has no NativeShaderExt: native shader work is "
                       "skipped and the DSL parts still run.",
                       backend);
        }
        ns::ShaderRegistry shaders;
        shaders.set_device(device, native_ext);
        register_builtin_kernels(device, shaders);

        // The stream must accept the GUI's graphics commands only when a
        // window will actually be shown: offline stays on a compute stream,
        // and so does an interactive run whose GUI is disabled or unavailable.
        auto interactive = document.mode.interactive;
        auto use_interactive_gui = interactive && document.mode.gui && !options.no_gui;
        auto stream = device.create_stream(use_interactive_gui ? StreamTag::GRAPHICS
                                                               : StreamTag::COMPUTE);
        auto dstorage_stream = Stream{};
        auto has_dstorage = create_dstorage_stream(device, document, dstorage_stream);
        if (!has_dstorage && document.config.dstorage.enabled) {
            LUISA_INFO("backend '{}' has no DStorageExt: resource inputs are read on "
                       "the host.",
                       backend);
        }
        if (!shaders.compile_all(document, {}, backend, native_ext, paths, diagnostics)) {
            report_diagnostics(diagnostics);
            return 1;
        }
        ns::ResourceRegistry resources;
        if (!resources.create_all(device, stream, has_dstorage ? &dstorage_stream : nullptr,
                                  document, paths, diagnostics)) {
            report_diagnostics(diagnostics);
            return 1;
        }
        for (auto &&entry : resources.entries()) {
            LUISA_INFO("resource '{}': {} ({} byte(s))", entry.spec.name,
                       ns::to_string(entry.spec.type), entry.resource.byte_size);
        }
        ns::WorkflowExecutor executor{device, stream,
                                      has_dstorage ? &dstorage_stream : nullptr,
                                      resources, shaders, paths, diagnostics};
        executor.set_sync_uploads(options.sync_uploads);
        if (interactive) {
#ifdef LUISA_ENABLE_GUI
            if (!document.mode.gui || options.no_gui) {
                report_failure("interactive mode needs the GUI: this build has no GUI "
                               "support or --no-gui was given; pass --offline (there is "
                               "no silent fallback)");
                return 1;
            }
            if (auto code = run_interactive(device, stream, options, document,
                                            resources, shaders, executor, paths);
                code != 0) {
                return code;
            }
#else
            report_failure("interactive mode is not available: this build has no GUI "
                           "support (rebuild with lc_enable_gui / "
                           "LUISA_COMPUTE_ENABLE_GUI, or pass --offline)");
            return 1;
#endif
        } else {
            Clock clock;
            auto frames = std::max(1u, document.mode.frames);
            for (auto frame = 0u; frame < frames; frame++) {
                // File sinks are written once, on the last frame, so that a
                // multi-frame run neither rewrites nor refuses to overwrite its
                // own output.
                executor.set_write_sinks(frame + 1u == frames);
                if (!executor.execute(document)) {
                    report_diagnostics(diagnostics);
                    return 1;
                }
            }
            if (options.stats) {
                LUISA_INFO("{} frame(s) of {} command(s) in {:.3f} ms ({:.3f} ms/frame)",
                           frames, executor.last_command_count(), clock.toc(),
                           clock.toc() / static_cast<double>(frames));
            }
            command_counts.assign(executor.last_command_counts().begin(),
                                  executor.last_command_counts().end());
            LUISA_INFO("dispatch document executed on '{}': {} frame(s), {} device "
                       "command(s), per-kind [{}]",
                       backend, frames, executor.last_command_count(),
                       summarize_command_counts(command_counts));
        }
        // The whole workflow is done — offline: every frame ran; interactive:
        // the window closed or the fixed frame count was reached. Write the
        // resources the document marked with `export_path` (a failed export is
        // a warning, not an error) and synchronize, so every device command
        // the run submitted is complete before main returns.
        ns::export_marked_resources(stream, resources, paths, diagnostics);
        stream << synchronize();
        if (!diagnostics.ok()) {
            report_diagnostics(diagnostics);
            return 1;
        }
        report_diagnostics(diagnostics);
        return 0;
    }();
    if (report != nullptr) {
        report->diagnostics = std::move(diagnostics);
        report->command_counts = std::move(command_counts);
    }
    return status;
}

// ---------------------------------------------------------------------------
// self test
// ---------------------------------------------------------------------------

struct Case {
    const char *name;
    std::string_view json;
    const char *expected_path;// "" == only require a rejection
};

// Documents that must be rejected, one per corner-case class. `expected_path` is
// the JSON path some diagnostic must name: the message wording is free to
// change, the path is the contract. Both stages the tool runs are exercised -
// the structural parse and the semantic pass over the parsed document.
constexpr Case kNegativeCases[] = {
    {"root not an object", "[]", ""},
    {"root is a scalar", "42", ""},
    {"truncated json", "{\"version\": 1,", ""},
    {"unsupported version", "{\"version\": 2}", "version"},
    {"version zero", "{\"version\": 0}", "version"},
    {"mode is not an object", "{\"mode\": 3}", "mode"},
    {"bad mode type", "{\"mode\": {\"type\": \"headless\"}}", "type"},
    {"zero offline frames", "{\"mode\": {\"type\": \"offline\", \"frames\": 0}}", "frames"},
    {"bad config type", "{\"config\": {\"optimize\": \"yes\"}}", "optimize"},
    {"bad limits value", "{\"config\": {\"limits\": {\"max_resources\": -1}}}", "max_resources"},
    {"resources not an array", "{\"resources\": {}}", "resources"},
    {"unknown resource type", "{\"resources\": [{\"name\": \"a\", \"type\": \"tensor\"}]}", "type"},
    {"illegal resource name",
     "{\"resources\": [{\"name\": \"a b\", \"type\": \"buffer\", \"byte_size\": 8}]}",
     "name"},
    {"duplicate resource name",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 8},"
     " {\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 8}]}",
     "name"},
    {"buffer count zero",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"element\": \"float\", \"count\": 0}]}",
     "count"},
    {"unknown buffer element",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"element\": \"float9\", \"count\": 4}]}",
     "element"},
    {"unknown pixel storage",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"texture\", \"storage\": \"float9\", \"size\": [4, 4]}]}",
     "storage"},
    {"block compressed storage",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"texture\", \"storage\": \"bc7\", \"size\": [4, 4]}]}",
     "storage"},
    {"zero texture extent",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"texture\", \"storage\": \"float4\", \"size\": [0, 4]}]}",
     "size"},
    {"odd hex payload",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 4,"
     " \"input\": {\"inline\": {\"hex\": \"abc\"}}}]}",
     "hex"},
    {"workflow not an array", "{\"workflow\": {}}", "workflow"},
    {"unknown cmd", "{\"workflow\": [{\"cmd\": \"explode\"}]}", "cmd"},
    {"cmd missing", "{\"workflow\": [{}]}", "cmd"},
    {"key not valid for cmd",
     "{\"workflow\": [{\"cmd\": \"log\", \"message\": \"x\", \"bindings\": []}]}",
     "bindings"},
    {"dangling resource",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_copy\", \"src\": \"a\", \"dst\": \"b\"}]}",
     "dst"},
    {"dangling shader",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"nope\","
     " \"dispatch\": [1, 1, 1]}]}",
     "shader"},
    // A missing binding is reported by `NativeShaderLauncher::validate()` at
    // dispatch time (the execution corpus covers that), so the validator only
    // rejects a *duplicate* one.
    {"duplicate native dispatch binding",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"element\": \"float\", \"count\": 4}],"
     " \"shaders\": [{\"name\": \"s\", \"source\": \"void main() {}\"}],"
     " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"dispatch\": [4, 1, 1],"
     " \"bindings\": [{\"index\": 0, \"resource\": \"a\", \"usage\": \"read\"},"
     " {\"index\": 0, \"resource\": \"a\", \"usage\": \"read\"}]}]}",
     "bindings"},
    {"native dispatch with dispatch and grid",
     "{\"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\","
     " \"dispatch\": [1, 1, 1], \"grid\": [1, 1, 1]}]}",
     "grid"},
    {"zero native dispatch",
     "{\"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\"}]}",
     ""},
    {"unknown uniform type",
     "{\"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"dispatch\": [1, 1, 1],"
     " \"uniforms\": [{\"type\": \"float9\", \"value\": 1.0}]}]}",
     "type"},
    {"bad sampler filter",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"modifications\": [{\"slot\": 0, \"kind\": \"texture2d\", \"resource\": \"i\","
     " \"sampler\": {\"filter\": \"nearest\", \"address\": \"repeat\"}}]}]}",
     "filter"},
    {"bindless slot out of range",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"modifications\": [{\"slot\": 9, \"kind\": \"buffer\", \"resource\": \"h\"}]}]}",
     "slot"},
    {"mesh stride zero",
     "{\"resources\": [{\"name\": \"m\", \"type\": \"mesh\", \"vertex_buffer\": \"v\","
     " \"triangle_buffer\": \"t\"}], \"workflow\": [{\"cmd\": \"mesh_build\","
     " \"resource\": \"m\", \"vertex_stride\": 0}]}",
     "vertex_stride"},
    {"accel update flag with modifications",
     "{\"resources\": [{\"name\": \"as\", \"type\": \"accel\"}],"
     " \"workflow\": [{\"cmd\": \"accel_build\", \"resource\": \"as\", \"instance_count\": 1,"
     " \"update_instance_buffer_only\": true, \"modifications\": [{\"index\": 0}]}]}",
     "modifications"},
    {"interactive without display image",
     "{\"mode\": {\"type\": \"interactive\"}}",
     "display_image"},
    {"interactive display scale zero",
     "{\"mode\": {\"type\": \"interactive\", \"display_image\": \"i\", \"display_scale\": 0}}",
     "display_scale"},
    {"sink without a path",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_download\", \"resource\": \"a\","
     " \"output\": {\"discard\": false}}]}",
     "output"},
    {"dangling verify source",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_download\", \"resource\": \"a\","
     " \"verify\": {\"kind\": \"copy\", \"source\": \"missing\"}}]}",
     "source"},
    {"bad shader language",
     "{\"shaders\": [{\"name\": \"s\", \"language\": \"rust\", \"source\": \"x\"}]}",
     "language"},
    {"duplicate shader name",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"a\"}, {\"name\": \"s\", \"source\": \"b\"}]}",
     "name"},
    // A name may be declared once per *language*: two entries that claim
    // the same one (or that both derive theirs at run time) are ambiguous.
    {"duplicate shader language",
     "{\"shaders\": [{\"name\": \"s\", \"language\": \"hlsl\", \"source\": \"a\"},"
     " {\"name\": \"s\", \"language\": \"hlsl\", \"source\": \"b\"}]}",
     "name"},
    // ---- manual reflection metadata (the "bindings" key of a shader) --------
    {"unknown shader binding kind",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\", \"bindings\": [{\"kind\": \"tensor\"}]}]}",
     "kind"},
    {"shader binding usage none",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\","
     " \"bindings\": [{\"kind\": \"structured_buffer\", \"usage\": \"none\"}]}]}",
     "usage"},
    {"read-only shader binding declared writable",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\","
     " \"bindings\": [{\"kind\": \"structured_buffer\", \"usage\": \"write\"}]}]}",
     "usage"},
    {"shader binding array size zero",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\","
     " \"bindings\": [{\"kind\": \"structured_buffer\", \"array_size\": 0}]}]}",
     "array_size"},
    {"duplicate shader binding address",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\","
     " \"bindings\": [{\"kind\": \"structured_buffer\", \"register\": 0},"
     " {\"kind\": \"rw_structured_buffer\", \"register\": 0}]}]}",
     "bindings"},
    // ---- parsing corner cases (section 7 of the plan) ----------------------
    {"trailing garbage after the document", "{\"version\": 1} trailing", ""},
    {"fractional value in an integer field", "{\"mode\": {\"frames\": 1.5}}", "frames"},
    {"negative count",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"element\": \"float\", \"count\": -1}]}",
     "count"},
    {"integer above the u32 range",
     "{\"mode\": {\"exit_after_frames\": 99999999999}}",
     "exit_after_frames"},
    {"number out of the double range", "{\"mode\": {\"display_scale\": 1e999}}", ""},
    {"byte size zero",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 0}]}",
     "byte_size"},
    {"zero mip levels",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4], \"levels\": 0}]}",
     "levels"},
    {"unknown usage",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\"}],"
     " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"dispatch\": [4, 1, 1],"
     " \"bindings\": [{\"index\": 0, \"resource\": \"a\", \"usage\": \"write_only\"}]}]}",
     "usage"},
    {"unknown compression",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16,"
     " \"input\": {\"file\": \"x.bin\", \"compression\": \"zstd\"}}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "compression"},
    {"unknown build request",
     "{\"resources\": [{\"name\": \"m\", \"type\": \"mesh\", \"vertex_buffer\": \"v\","
     " \"triangle_buffer\": \"t\"}],"
     " \"workflow\": [{\"cmd\": \"mesh_build\", \"resource\": \"m\", \"request\": \"maybe\"}]}",
     "request"},
    {"unknown sampler address",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"modifications\": [{\"slot\": 0, \"kind\": \"texture2d\", \"resource\": \"i\","
     " \"sampler\": {\"filter\": \"linear_linear\", \"address\": \"clamp\"}}]}]}",
     "address"},
    {"unknown bindless mode",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"mode\": \"sampler\"}]}",
     "mode"},
    {"unknown bindless operation",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"modifications\": [{\"slot\": 0, \"kind\": \"buffer\", \"op\": \"append\","
     " \"resource\": \"h\"}]}]}",
     "op"},
    {"bindless slot type disagrees with the modification",
     "{\"resources\": [{\"name\": \"h\", \"type\": \"bindless_array\", \"slot_count\": 4,"
     " \"slot_type\": \"buffer\"}],"
     " \"workflow\": [{\"cmd\": \"bindless_array_update\", \"resource\": \"h\","
     " \"modifications\": [{\"slot\": 0, \"kind\": \"texture2d\", \"resource\": \"h\"}]}]}",
     "modifications"},
    {"textured storage disagrees with the resource",
     "{\"resources\": [{\"name\": \"t\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}],"
     " \"workflow\": [{\"cmd\": \"texture_download\", \"resource\": \"t\", \"storage\": \"byte4\","
     " \"output\": {\"discard\": true}}]}",
     "storage"},
    {"wrong resource kind for the command",
     "{\"resources\": [{\"name\": \"t\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}],"
     " \"workflow\": [{\"cmd\": \"buffer_copy\", \"src\": \"t\", \"dst\": \"t\"}]}",
     "src"},
    {"zero window size",
     "{\"mode\": {\"type\": \"interactive\", \"display_image\": \"i\","
     " \"window\": {\"width\": 0}},"
     " \"resources\": [{\"name\": \"i\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}]}",
     "window"},
    {"display image is not a 4-channel float image",
     "{\"mode\": {\"type\": \"interactive\", \"display_image\": \"i\"},"
     " \"resources\": [{\"name\": \"i\", \"type\": \"texture\", \"storage\": \"float1\","
     " \"size\": [4, 4]}]}",
     "display_image"},
    {"display image is not a texture",
     "{\"mode\": {\"type\": \"interactive\", \"display_image\": \"b\"},"
     " \"resources\": [{\"name\": \"b\", \"type\": \"buffer\", \"byte_size\": 64}]}",
     "display_image"},
    {"a binding without a resource",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\"}],"
     " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"grid\": [1, 1, 1],"
     " \"bindings\": [{\"index\": 0, \"usage\": \"read\"}]}]}",
     "bindings"},
    {"a binding without a usage",
     "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\"}],"
     " \"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"grid\": [1, 1, 1],"
     " \"bindings\": [{\"index\": 0, \"resource\": \"a\", \"usage\": \"none\"}]}]}",
     "usage"},
    {"a resource over the byte budget",
     "{\"config\": {\"limits\": {\"max_resource_bytes\": 1024}},"
     " \"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 4096}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "max_resource_bytes"},
    // The names this example deliberately does not implement: curve and
    // motion-blur acceleration structures (no backend implements them) and
    // indirect dispatch (unsupported everywhere). The codec must reject them with
    // a message saying why, not with a generic "unknown name".
    {"retired cmd: curve_build",
     "{\"workflow\": [{\"cmd\": \"curve_build\", \"resource\": \"c\"}]}",
     "cmd"},
    {"retired cmd: motion_instance_build",
     "{\"workflow\": [{\"cmd\": \"motion_instance_build\", \"resource\": \"i\"}]}",
     "cmd"},
    {"retired resource type: curve",
     "{\"resources\": [{\"name\": \"c\", \"type\": \"curve\"}]}",
     "type"},
    {"retired resource type: motion_instance",
     "{\"resources\": [{\"name\": \"i\", \"type\": \"motion_instance\"}]}",
     "type"},
    {"retired resource type: indirect_dispatch_buffer",
     "{\"resources\": [{\"name\": \"i\", \"type\": \"indirect_dispatch_buffer\", \"capacity\": 8}]}",
     "type"},
    {"retired indirect dispatch form",
     "{\"resources\": [{\"name\": \"i\", \"type\": \"indirect_dispatch_buffer\", \"capacity\": 8}],"
     " \"workflow\": [{\"cmd\": \"shader_dispatch\", \"shader\": \"scale_buffer\","
     " \"arguments\": [{\"kind\": \"buffer\", \"resource\": \"i\"}],"
     " \"indirect\": {\"resource\": \"i\", \"offset\": 0, \"max_dispatch_size\": 64}}]}",
     "indirect"},

};

// Documents that must still be accepted (parse + semantic pass clean) but that
// must produce a warning naming the path: an unknown key is a warning, not an
// error, unless `--strict` escalates it (which native_shader.cpp does, so the
// corpus also covers the warning path).
constexpr Case kWarningCases[] = {
    {"unknown top level key", "{\"version\": 1, \"nope\": 1}", "nope"},
    {"unknown mode key", "{\"mode\": {\"type\": \"offline\", \"zap\": 1}}", "zap"},
    {"unknown config key", "{\"config\": {\"nope\": 1}}", "nope"},
    {"duplicate key", "{\"version\": 1, \"version\": 1}", "version"},
    {"unreferenced resource",
     "{\"resources\": [{\"name\": \"lonely_buffer\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "lonely_buffer"},
    {"offline run with an empty workflow", "{\"workflow\": []}", "workflow"},
    {"a typo in a key name is a warning, not an error",
     "{\"resources\": [{\"name\": \"t\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}, {\"name\": \"b\", \"type\": \"buffer\", \"byte_size\": 64}],"
     " \"workflow\": [{\"cmd\": \"buffer_to_texture_copy\", \"buffer\": \"b\", \"texture\": \"t\","
     " \"storage\": \"float4\", \"size\": [4, 4, 1], \"textureoffset\": [0, 0, 0]}]}",
     "textureoffset"},
    {"a resource that shadows a shader name",
     "{\"shaders\": [{\"name\": \"shadowed\", \"source\": \"x\"}],"
     " \"resources\": [{\"name\": \"shadowed\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "shadowed"},
    {"per-frame dispatch while the workflow downloads",
     "{\"mode\": {\"type\": \"interactive\", \"display_image\": \"i\"},"
     " \"resources\": [{\"name\": \"i\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}, {\"name\": \"sink\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_download\", \"resource\": \"sink\","
     " \"output\": {\"discard\": true}}]}",
     "dispatch_per_frame"},
    // `export_path` is only a known key of buffers, textures and volumes: on
    // any other resource type it reads as an unknown key (a warning, an
    // error under --strict), never as a silently ignored export request.
    {"export_path on a resource that cannot be exported",
     "{\"resources\": [{\"name\": \"as\", \"type\": \"accel\","
     " \"export_path\": \"as.bin\"}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "export_path"},
};

// Rejects a document through the same two stages the tool runs, and returns the
// diagnostics the rejection produced.
[[nodiscard]] luisa::vector<luisa::string>
reject(const std::string_view json, luisa::vector<luisa::string> &warnings) noexcept {
    auto parsed = ns::parse_dispatch_json(json, ns::JsonLimits{});
    auto errors = luisa::vector<luisa::string>{};
    if (!parsed.value.has_value()) {
        for (auto &&error : parsed.errors) { errors.emplace_back(std::move(error)); }
        return errors;
    }
    for (auto &&warning : parsed.warnings) { warnings.emplace_back(std::move(warning)); }
    auto warnings_of_semantic_pass = luisa::vector<luisa::string>{};
    if (!ns::validate_dispatch_semantics(*parsed.value, parsed.value->config.limits,
                                         errors, warnings_of_semantic_pass) &&
        errors.empty()) {
        errors.emplace_back("the semantic validator rejected the document without a diagnostic");
    }
    for (auto &&warning : warnings_of_semantic_pass) {
        warnings.emplace_back(std::move(warning));
    }
    return errors;
}

[[nodiscard]] bool names_path(luisa::span<const luisa::string> diagnostics,
                              const char *path) noexcept {
    for (auto &&diagnostic : diagnostics) {
        if (diagnostic.find(path) != luisa::string::npos) { return true; }
    }
    return false;
}

// Resolves `relative` (a path under examples/compute/) against the candidate
// working directories a developer or CI runs the example from.
[[nodiscard]] luisa::filesystem::path find_example_file(luisa::string_view relative) noexcept {
    auto relative_path = luisa::filesystem::path{};
    if (!luisa::path_from_narrow(relative, relative_path)) { return {}; }
    constexpr const char *kRoots[] = {".", "..", "../..", "../../.."};
    std::error_code ec;
    for (auto root : kRoots) {
        auto candidate = luisa::filesystem::path{root} / "examples" / "compute" / relative_path;
        if (luisa::filesystem::is_regular_file(candidate, ec)) { return candidate; }
    }
    if (luisa::filesystem::is_regular_file(relative_path, ec)) { return relative_path; }
    return {};
}

[[nodiscard]] std::string_view embedded_document(luisa::string_view name) noexcept {
    auto path = luisa::format("native_shader_examples/{}", name);
    return ns::embedded_file(path);
}

// The codec's validation tables and the runtime's name -> device-type tables
// list the same names, so the self test walks one against the other.
[[nodiscard]] bool check_table_drift() noexcept {
    auto ok = true;
    for (auto name : ns::buffer_element_spellings()) {
        auto element = ns::BufferElement{};
        if (!ns::parse_buffer_element(name, element)) {
            report_failure(luisa::format("drift: the codec accepts the buffer element '{}' but "
                                         "the runtime cannot map it to a device type",
                                         name));
            ok = false;
        }
    }
    for (auto name : ns::pixel_storage_spellings()) {
        auto storage = PixelStorage{};
        if (!ns::parse_pixel_storage(name, storage)) {
            report_failure(luisa::format("drift: the codec accepts the pixel storage '{}' but "
                                         "the runtime cannot map it to a PixelStorage",
                                         name));
            ok = false;
        }
    }
    for (auto name : ns::usage_spellings()) {
        auto usage = Usage::NONE;
        if (!ns::parse_usage(name, usage)) {
            report_failure(luisa::format("drift: the codec accepts the usage '{}' but the "
                                         "runtime cannot map it to a Usage",
                                         name));
            ok = false;
        }
    }
    return ok;
}

// A document that needs no native shader at all: a host upload, a buffer copy, a
// DSL dispatch and a verified readback. This is the case that still runs on a
// backend without `NativeShaderExt`.
[[nodiscard]] luisa::string dsl_only_document() noexcept {
    auto bytes = luisa::vector<std::byte>(64u * sizeof(float));
    for (auto i = 0u; i < 64u; i++) {
        auto value = static_cast<float>(i);
        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
    }
    return luisa::format(
        R"({{"version":1,"mode":{{"type":"offline"}},)"
        R"("resources":[{{"name":"a","type":"buffer","element":"float","count":64,)"
        R"("input":{{"inline":{{"hex":"{}"}}}}}},)"
        R"({{"name":"b","type":"buffer","element":"float","count":64,)"
        R"("export_path":"dsl_only_b.bin"}}],)"
        R"("workflow":[)"
        R"({{"cmd":"buffer_copy","src":"a","dst":"b","size":256}},)"
        R"({{"cmd":"shader_dispatch","shader":"scale_buffer",)"
        R"("arguments":[{{"kind":"buffer","resource":"b"}},)"
        R"({{"kind":"buffer","resource":"b"}},)"
        R"({{"kind":"uniform","type":"float32","value":1.0}},)"
        R"({{"kind":"uniform","type":"float32","value":0.0}}],)"
        R"("dispatch":[64,1,1]}},)"
        R"({{"cmd":"buffer_download","resource":"b","output":{{"discard":true}},)"
        R"("verify":{{"kind":"linear","source":"a","k":1.0,"c":0.0}}}}]}})",
        ns::encode_hex(luisa::span<const std::byte>{bytes.data(), bytes.size()}));
}

// A document that parses and validates but must still be rejected when it runs:
// a device-side contract violation has to surface as a non-zero exit with a
// diagnostic, never as a crash (a crash would kill this process, which is the
// assertion). `expected_text`, when set, has to appear in a diagnostic.
struct RejectionCase {
    const char *name;
    std::string_view json;
    const char *expected_text;
};

constexpr RejectionCase kRejectionCases[] = {
    {"a buffer region past the end of the resource",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16},"
     " {\"name\": \"b\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_copy\", \"src\": \"a\", \"dst\": \"b\","
     " \"src_offset\": 0, \"dst_offset\": 0, \"size\": 1024}]}",
     ""},
    {"a texture region outside the mip",
     "{\"resources\": [{\"name\": \"t\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}, {\"name\": \"u\", \"type\": \"texture\", \"storage\": \"float4\","
     " \"size\": [4, 4]}],"
     " \"workflow\": [{\"cmd\": \"texture_copy\", \"storage\": \"float4\", \"src\": \"t\","
     " \"dst\": \"u\", \"size\": [64, 64, 1]}]}",
     ""},
    {"a mesh vertex stride that does not divide the buffer",
     "{\"resources\": [{\"name\": \"v\", \"type\": \"buffer\", \"element\": \"float\", \"count\": 5},"
     " {\"name\": \"t\", \"type\": \"buffer\", \"element\": \"triangle\", \"count\": 1},"
     " {\"name\": \"m\", \"type\": \"mesh\", \"vertex_buffer\": \"v\","
     " \"triangle_buffer\": \"t\"}],"
     " \"workflow\": [{\"cmd\": \"mesh_build\", \"resource\": \"m\", \"vertex_buffer\": \"v\","
     " \"vertex_buffer_offset\": 0, \"vertex_buffer_size\": 20, \"vertex_stride\": 8,"
     " \"triangle_buffer\": \"t\", \"triangle_buffer_offset\": 0,"
     " \"triangle_buffer_size\": 12}]}",
     ""},
    {"a shader_dispatch with the wrong argument count",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"element\": \"float\","
     " \"count\": 64}],"
     " \"workflow\": [{\"cmd\": \"shader_dispatch\", \"shader\": \"scale_buffer\","
     " \"arguments\": [{\"kind\": \"buffer\", \"resource\": \"a\"}],"
     " \"dispatch\": [64, 1, 1]}]}",
     "argument"},
    {"an input file that does not exist",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16,"
     " \"input\": {\"file\": \"no_such_input_file.bin\"}}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "no_such_input_file.bin"},
    {"an input region past the end of the file",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16,"
     " \"input\": {\"file\": \"selftest_src.bin\", \"offset\": 65536, \"size\": 16}}],"
     " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
     "selftest_src.bin"},
    {"a PNG sink that is not a 2-D byte4/float4 image",
     "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16}],"
     " \"workflow\": [{\"cmd\": \"buffer_download\", \"resource\": \"a\","
     " \"output\": {\"file\": \"nope.png\", \"format\": \"png\", \"overwrite\": true}}]}",
     ""},
      {"an unknown custom command uuid",
       "{\"workflow\": [{\"cmd\": \"custom_command\", \"uuid\": 4660}]}",
       "uuid"},
      {"a native shader that does not compile",
       // One broken variant per language: every backend compiles the variant it
       // speaks and must refuse the document with the compiler's diagnostic,
       // never a crash or a hang.
       "{\"shaders\": [{\"name\": \"broken\", \"language\": \"hlsl\","
       " \"source_type\": \"code\", \"entry_point\": \"CSMain\","
       " \"block_size\": [32, 1, 1], \"source\": \"RWStructuredBuffer<float> buf"
       " : register(u0);\\n[numthreads(32, 1, 1)]\\nvoid CSMain(uint3 tid :"
       " SV_DispatchThreadID) { buf[tid.x] = tid.x +; }\"},"
       " {\"name\": \"broken\", \"language\": \"glsl\", \"source_type\": \"code\","
       " \"entry_point\": \"main\", \"block_size\": [32, 1, 1], \"source\": \"#version"
       " 450\\nlayout(local_size_x = 32) in;\\nlayout(set = 0, binding = 0, std430)"
       " buffer D { float v[]; } b;\\nvoid main() { b.v[gl_GlobalInvocationID.x] ="
       " 1.0 +; }\"},"
       " {\"name\": \"broken\", \"language\": \"cuda_nvrtc\", \"source_type\": \"code\","
       " \"entry_point\": \"broken\", \"block_size\": [32, 1, 1], \"source\": \"extern"
       " \\\"C\\\" __global__ void broken(float *buf) { auto i = blockIdx.x *"
       " blockDim.x + threadIdx.x; buf[i] = 1.0f +; }\"}],"
       " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
       "compilation failed"},
};
[[nodiscard]] int run_self_test(Context &context, const Options &options) noexcept {
    auto checks = 0u;
    auto failures = 0u;
    auto check = [&](bool condition, const luisa::string &what) noexcept {
        checks++;
        if (!condition) {
            failures++;
            report_failure(what);
        }
    };

    // ---- 0. the codec and the runtime must agree on the name tables ---------
    check(check_table_drift(),
          "drift: the codec's spelling tables and the runtime's mappings disagree");
    // ---- 1. the embedded corpus must match the files it names --------------
    // Translation phase 1 maps every source newline to '\n', so a raw string
    // compiled into the binary always holds LF even when the on-disk twin was
    // checked out with CRLF endings: compare with the line endings normalized
    // or every corpus entry reads as "stale" on such a checkout.
    auto normalize_newlines = [](std::string_view text) noexcept {
        auto out = luisa::string{};
        out.reserve(text.size());
        for (auto i = size_t{0u}; i < text.size(); i++) {
            if (text[i] == '\r' && i + 1u < text.size() && text[i + 1u] == '\n') { continue; }
            out.push_back(text[i]);
        }
        return out;
    };
    auto missing_files = 0u;
    for (auto &&file : ns::kEmbeddedFiles) {
        auto path = find_example_file(file.path);
        if (path.empty()) {
            missing_files++;
            continue;
        }
        auto bytes = luisa::vector<std::byte>{};
        if (auto error = luisa::string{}; !ns::read_file(path, 1u << 24u, bytes, error)) {
            check(false, luisa::format("corpus: cannot read '{}': {}", file.path, error));
            continue;
        }
        auto text = std::string_view{reinterpret_cast<const char *>(bytes.data()),
                                     bytes.size()};
        check(normalize_newlines(text) == normalize_newlines(file.contents),
              luisa::format("corpus: the embedded copy of '{}' is stale", file.path));
    }
    if (missing_files != 0u) {
        LUISA_INFO("self test: {} embedded file(s) have no on-disk twin here; the "
                   "textual cross-check was skipped for them.",
                   missing_files);
    }
    // ---- 2. codec round trip ----------------------------------------------
    const std::string_view documents[] = {
        ns::kEmbeddedDefaultDocument,
        embedded_document("scale_offline.json"),
        embedded_document("scale_interactive.json"),
        embedded_document("all_commands_offline.json"),
    };
    for (auto document : documents) {
        auto first = ns::parse_dispatch_json(document, ns::JsonLimits{});
        check(first.value.has_value(),
              luisa::format("round trip: a corpus document does not parse ({})",
                            first.errors.empty() ? "no diagnostic" : first.errors.front()));
        if (!first.value.has_value()) { continue; }
        auto written = ns::write_dispatch_json(*first.value);
        check(written.error.empty(),
              luisa::format("round trip: serialization failed: {}", written.error));
        if (!written.error.empty()) { continue; }
        auto second = ns::parse_dispatch_json(written.json, ns::JsonLimits{});
        check(second.value.has_value(), "round trip: the written document does not parse");
        if (!second.value.has_value()) { continue; }
        auto rewritten = ns::write_dispatch_json(*second.value);
        check(rewritten.json == written.json,
              "round trip: write(read(write(document))) is not byte-stable");
        auto mismatch = luisa::string{};
        check(ns::equivalent(*first.value, *second.value, mismatch),
              luisa::format("round trip: the documents differ: {}", mismatch));
    }
    // ---- 3. negative and warning corpora -----------------------------------
    for (auto &&negative : kNegativeCases) {
        auto warnings = luisa::vector<luisa::string>{};
        auto errors = reject(negative.json, warnings);
        check(!errors.empty(), luisa::format("negative corpus: '{}' was accepted", negative.name));
        if (errors.empty() || negative.expected_path[0] == '\0') { continue; }
        check(names_path(errors, negative.expected_path),
              luisa::format("negative corpus: '{}' does not name the path '{}' "
                            "(first diagnostic: '{}')",
                            negative.name, negative.expected_path, errors.front()));
    }
    for (auto &&warning_case : kWarningCases) {
        auto warnings = luisa::vector<luisa::string>{};
        auto errors = reject(warning_case.json, warnings);
        check(errors.empty(), luisa::format(
                                  "warning corpus: '{}' was rejected ('{}')",
                                  warning_case.name,
                                  errors.empty() ? "no diagnostic" : errors.front()));
        check(names_path(warnings, warning_case.expected_path),
              luisa::format("warning corpus: '{}' does not warn about the path '{}'",
                            warning_case.name, warning_case.expected_path));
    }
    // ---- 3b. cases that are easier to build than to spell -------------------
    // A document nested past `max_depth`.
    {
        auto deep = luisa::string{"{\"version\": 1, \"mode\": "};
        for (auto i = 0u; i < 80u; i++) { deep.push_back('['); }
        for (auto i = 0u; i < 80u; i++) { deep.push_back(']'); }
        deep.push_back('}');
        auto parsed = ns::parse_dispatch_json(deep, ns::JsonLimits{});
        check(!parsed.value.has_value(),
              "negative corpus: a document nested past max_depth was accepted");
    }
    // A UTF-8 BOM is accepted, which is what `YYJSON_READ_ALLOW_BOM` is for.
    {
        auto bom = luisa::string{};
        bom.push_back(static_cast<char>(0xEF));
        bom.push_back(static_cast<char>(0xBB));
        bom.push_back(static_cast<char>(0xBF));
        bom.append("{\"version\": 1, \"workflow\": [{\"cmd\": \"log\", \"message\": \"bom\"}]}");
    auto parsed = ns::parse_dispatch_json(bom, ns::JsonLimits{});
    check(parsed.value.has_value(), "codec: a document with a UTF-8 BOM was rejected");
    }
    // A binding with no `index` and no `register`/`space` is the positional form
    // the launcher fills in reflection order: several of them are legal and must
    // not be read as "the same register twice".
    {
        auto warnings = luisa::vector<luisa::string>{};
        auto errors = reject(
            "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\"}],"
            " \"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16},"
            "  {\"name\": \"b\", \"type\": \"buffer\", \"byte_size\": 16}],"
            " \"workflow\": [{\"cmd\": \"native_dispatch\", \"shader\": \"s\", \"grid\": [1, 1, 1],"
            "  \"bindings\": [{\"resource\": \"a\", \"usage\": \"read\"},"
            "   {\"resource\": \"b\", \"usage\": \"write\"}]}]}",
            warnings);
        check(errors.empty(), luisa::format(
                                  "codec: positional bindings were rejected ('{}')",
                                  errors.empty() ? "no diagnostic" : errors.front()));
    }
    // A shader entry that declares its reflection manually ("bindings")
    // round-trips through the writer byte-stable.
    {
        auto text = std::string_view{
            "{\"shaders\": [{\"name\": \"s\", \"source\": \"x\","
            " \"bindings\": [{\"kind\": \"structured_buffer\", \"register\": 0,"
            " \"usage\": \"read\"},"
            " {\"kind\": \"rw_structured_buffer\", \"space\": 0, \"register\": 1,"
            " \"array_size\": 2}]}] }"};
        auto first = ns::parse_dispatch_json(text, ns::JsonLimits{});
        check(first.value.has_value() && first.value->shaders.size() == 1u &&
                  first.value->shaders[0].bindings.size() == 2u,
              "codec: a shader with manual reflection metadata was rejected");
        if (first.value.has_value()) {
            auto written = ns::write_dispatch_json(*first.value);
            check(written.error.empty(),
                  luisa::format("codec: serialising manual reflection failed: {}",
                                written.error));
            auto second = ns::parse_dispatch_json(written.json, ns::JsonLimits{});
            check(second.value.has_value(),
                  "codec: the written manual reflection does not parse");
            auto mismatch = luisa::string{};
            check(second.value.has_value() &&
                      ns::equivalent(*first.value, *second.value, mismatch),
                  luisa::format("codec: the manual reflection does not round-trip: {}",
                                mismatch));
        }
    }
    // A resource `export_path` survives the write/read cycle: the document
    // below marks one buffer and one texture for the end-of-run export.
    {
        auto text = std::string_view{
            "{\"resources\": [{\"name\": \"b\", \"type\": \"buffer\", \"byte_size\": 16,"
            " \"export_path\": \"out/b.bin\"},"
            " {\"name\": \"t\", \"type\": \"texture\", \"storage\": \"float4\","
            " \"size\": [4, 4], \"export_path\": \"out/t.raw\"}],"
            " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}"};
        auto first = ns::parse_dispatch_json(text, ns::JsonLimits{});
        check(first.value.has_value() && first.value->resources.size() == 2u &&
                  first.value->resources[0].export_path == "out/b.bin" &&
                  first.value->resources[1].export_path == "out/t.raw",
              "codec: resource 'export_path' fields were rejected or lost");
        if (first.value.has_value()) {
            auto written = ns::write_dispatch_json(*first.value);
            check(written.error.empty(),
                  luisa::format("codec: serialising 'export_path' failed: {}",
                                written.error));
            auto second = ns::parse_dispatch_json(written.json, ns::JsonLimits{});
            auto mismatch = luisa::string{};
            check(second.value.has_value() &&
                      ns::equivalent(*first.value, *second.value, mismatch),
                  luisa::format("codec: 'export_path' does not round-trip: {}", mismatch));
        }
    }

    // ---- 4. execution corpus ----------------------------------------------
    auto backend = options.backend;
    auto backend_is_hlsl = backend == "dx" || backend == "vk";

    // Native shader support decides which execution cases can run at all: a
    // backend without `NativeShaderExt` still exercises the codec, the registered
    // DSL kernels and the host data paths, and reports the skip instead of
    // failing.
    auto native_shaders_supported = false;
    if (auto probe = context.create_device(backend); probe) {
        native_shaders_supported = probe.extension<NativeShaderExt>() != nullptr;
    }
    if (!native_shaders_supported) {
        LUISA_INFO("self test: backend '{}' has no NativeShaderExt, so the "
                   "shader-dependent execution cases are skipped (the codec, the "
                   "registered DSL kernels and the host data paths stay covered).",
                   backend);
    }
    if (native_shaders_supported) {
        // 4a. the embedded default workflow on this backend
        {
            auto document = ns::DispatchJson{};
            if (auto build_error = luisa::string{}; !build_default_document(backend, options, document, build_error)) {
                check(false, luisa::format("execution: {}", build_error));
            } else {
                auto errors = luisa::vector<luisa::string>{};
                auto warnings = luisa::vector<luisa::string>{};
                check(ns::validate_dispatch_semantics(document, document.config.limits, errors, warnings),
                      luisa::format("execution: the default document does not validate: {}",
                                    errors.empty() ? "no diagnostic" : errors.front()));
                auto status = run_document(context, backend, options, std::move(document), ".");
                check(status == 0, luisa::format(
                                       "execution: the default workflow failed on '{}' (exit {})",
                                       backend, status));
            }
        }
        // 4b. a document whose input comes from a file: this is what exercises the
        //     dstorage path (or its host fallback) on the backends that have one.
        {
            auto document = ns::DispatchJson{};
            if (auto build_error = luisa::string{}; !build_default_document(backend, options, document, build_error)) {
                check(false, luisa::format("execution: {}", build_error));
            } else {
                auto directory = luisa::filesystem::path{"."};
                std::error_code ec;
                luisa::filesystem::create_directories(directory, ec);
                auto bytes = luisa::vector<std::byte>(64u * sizeof(float));
                for (auto i = 0u; i < 64u; i++) {
                    auto value = static_cast<float>(i);
                    std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
                }
                auto file = directory / "native_shader_output" / "selftest_src.bin";
                if (auto error = luisa::string{}; !ns::write_output(
                        file, luisa::span<const std::byte>{bytes.data(), bytes.size()}, "raw",
                        uint3{0u, 0u, 0u}, PixelStorage::BYTE1, true, error)) {
                    check(false, luisa::format("execution: cannot write the self-test input: {}", error));
                } else {
                    document.resources.front().input.kind = ns::InputJson::Kind::File;
                    document.resources.front().input.file = "selftest_src.bin";
                    document.resources.front().input.offset = 0u;
                    document.resources.front().input.size = bytes.size();
                    auto status = run_document(context, backend, options, std::move(document),
                                               luisa::to_string(directory / "native_shader_output"));
                    check(status == 0, luisa::format(
                                           "execution: the file-input workflow failed on '{}' "
                                           "(exit {})",
                                           backend, status));
                }
            }
        }
        // 4c. the all-commands corpus needs an HLSL-capable backend and RTX.
        if (backend_is_hlsl) {
            auto sample_dir = find_example_file("native_shader_examples/scale_offline.json");
            if (sample_dir.empty()) {
                LUISA_INFO("self test: the sample directory was not found; the "
                           "all-commands corpus was skipped.");
            } else {
                auto directory = luisa::to_string(sample_dir.parent_path());
                auto corpus = luisa::filesystem::path{};
                if (!luisa::path_from_narrow(directory, corpus)) {
                    check(false, luisa::format("execution: the corpus directory '{}' is not "
                                               "usable",
                                               directory));
                    corpus = luisa::filesystem::path{"."};
                }
                auto parsed = ns::parse_dispatch_file(corpus / "all_commands_offline.json",
                                                      ns::JsonLimits{});
                check(parsed.value.has_value(),
                      luisa::format("execution: the all-commands corpus does not parse: {}",
                                    parsed.errors.empty() ? "no diagnostic" : parsed.errors.front()));
                if (parsed.value.has_value()) {
                    auto status = run_document(context, backend, options,
                                               std::move(*parsed.value), directory);
                    check(status == 0, luisa::format(
                                           "execution: the all-commands corpus failed on '{}' "
                                           "(exit {})",
                                           backend, status));
                }
            }
        } else {
            LUISA_INFO("self test: '{}' does not accept HLSL, so the all-commands corpus "
                       "was skipped.",
                       backend);
        }
    }

    // 4e. the DSL-only document runs everywhere: it needs no native shader, so it
    //     is the one execution case a backend without `NativeShaderExt` still
    //     covers end to end.
    {
        auto parsed = ns::parse_dispatch_json(dsl_only_document(), ns::JsonLimits{});
        check(parsed.value.has_value(), "execution: the DSL-only document does not parse");
        if (parsed.value.has_value()) {
            auto report = RunReport{};
            auto status = run_document(context, backend, options, std::move(*parsed.value),
                                       ".", &report);
            check(status == 0, luisa::format(
                                   "execution: the DSL-only document failed on '{}' (exit {})",
                                   backend, status));
            auto kind = static_cast<size_t>(luisa::to_underlying(ns::CommandKind::ShaderDispatch));
            check(kind < report.command_counts.size() && report.command_counts[kind] == 1u,
                  "execution: the DSL-only document did not run its shader_dispatch");
            // The document marks 'b' with `export_path`: after the workflow the
            // run must have written the resource to the output directory.
            auto exported = luisa::vector<std::byte>{};
            auto read_error = luisa::string{};
            auto export_file = luisa::filesystem::path{"."} / "native_shader_output" /
                               "dsl_only_b.bin";
            check(ns::read_file(export_file, 1u << 20u, exported, read_error) &&
                      exported.size() == 64u * sizeof(float),
                  luisa::format("execution: the exported resource is missing or has the "
                                "wrong size ('{}': {})",
                                luisa::to_string(export_file), read_error));
            if (exported.size() == 64u * sizeof(float)) {
                auto correct = true;
                for (auto i = 0u; i < 64u && correct; i++) {
                    auto value = 0.0f;
                    std::memcpy(&value, exported.data() + i * sizeof(float), sizeof(float));
                    correct = value == static_cast<float>(i);
                }
                check(correct, "execution: the exported resource holds the wrong data");
            }
        }
    }

    // 4f. an export that cannot be written downgrades to a warning: the run
    //     itself succeeded, so the exit code stays 0 and the diagnostic names
    //     the resource.
    {
        // A regular file sitting where the export wants a directory makes the
        // write fail for a reason the run cannot fix.
        auto blocker = luisa::filesystem::path{"."} / "native_shader_output" /
                       "export_blocker";
        if (auto error = luisa::string{}; !ns::write_output(
                blocker, luisa::span<const std::byte>{}, "raw", uint3{0u, 0u, 0u},
                PixelStorage::BYTE1, true, error)) {
            check(false, luisa::format("execution: cannot prepare the export blocker: {}",
                                       error));
        } else {
            auto parsed = ns::parse_dispatch_json(
                "{\"resources\": [{\"name\": \"a\", \"type\": \"buffer\", \"byte_size\": 16,"
                " \"export_path\": \"export_blocker/out.bin\"}],"
                " \"workflow\": [{\"cmd\": \"log\", \"message\": \"x\"}]}",
                ns::JsonLimits{});
            check(parsed.value.has_value(),
                  "execution: the failed-export document does not parse");
            if (parsed.value.has_value()) {
                auto report = RunReport{};
                auto status = run_document(context, backend, options,
                                           std::move(*parsed.value), ".", &report);
                check(status == 0,
                      luisa::format("execution: a failed export failed the run (exit {})",
                                    status));
                auto warned = false;
                for (auto &&warning : report.diagnostics.warnings) {
                    if (warning.find("export of resource 'a'") != luisa::string::npos) {
                        warned = true;
                        break;
                    }
                }
                check(warned,
                      "execution: a failed export did not produce a warning naming the "
                      "resource");
            }
            std::error_code ec;
            luisa::filesystem::remove(blocker, ec);
        }
    }

    // ---- 5. documents that must be rejected when they run -------------------
    // The file-based rejections need a file to point at; it is (re)written here so
    // the corpus does not depend on the order of the cases above.
    {
        auto bytes = luisa::vector<std::byte>(256u, std::byte{0u});
        auto path = luisa::filesystem::path{"native_shader_output"} / "selftest_src.bin";
        if (auto error = luisa::string{}; !ns::write_output(
                path, luisa::span<const std::byte>{bytes.data(), bytes.size()}, "raw",
                uint3{0u, 0u, 0u}, PixelStorage::BYTE1, true, error)) {
            LUISA_INFO("self test: cannot (re)write the self-test input file: {}", error);
        }
    }
    for (auto &&rejection : kRejectionCases) {
        auto warnings = luisa::vector<luisa::string>{};
        auto diagnostics = reject(rejection.json, warnings);
        auto rejected = !diagnostics.empty();
        if (!rejected) {
            // The document validator is happy, so the device has to reject it.
            auto parsed = ns::parse_dispatch_json(rejection.json, ns::JsonLimits{});
            if (!parsed.value.has_value()) {
                for (auto &&error : parsed.errors) { diagnostics.emplace_back(std::move(error)); }
                rejected = !diagnostics.empty();
            } else {
                auto report = RunReport{};
                g_quiet = true;
                auto status = run_document(context, backend, options, std::move(*parsed.value),
                                           "native_shader_output", &report);
                g_quiet = false;
                for (auto &&error : report.diagnostics.errors) {
                    diagnostics.emplace_back(std::move(error));
                }
                rejected = status != 0;
            }
        }
        check(rejected, luisa::format("rejection corpus: '{}' was accepted", rejection.name));
        if (rejection.expected_text[0] == '\0' || !rejected || diagnostics.empty()) { continue; }
        check(names_path(diagnostics, rejection.expected_text),
              luisa::format("rejection corpus: '{}' does not report '{}' "
                            "(first diagnostic: '{}')",
                            rejection.name, rejection.expected_text, diagnostics.front()));
    }

    LUISA_INFO("self test on '{}': {} check(s), {} failure(s).", backend, checks, failures);
    return failures == 0u ? 0 : 1;
}

}// namespace

int main(int argc, char *argv[]) {
    auto program = argc > 0 ? luisa::string_view{argv[0]} : luisa::string_view{"example_native_shader"};
    // The command line without the program name: `[backend, document, shaders..., options...]`.
    auto arguments = luisa::vector<luisa::string_view>{};
    arguments.reserve(argc > 0 ? static_cast<size_t>(argc - 1) : 0u);
    for (auto i = 1; i < argc; i++) { arguments.emplace_back(argv[i]); }
    // `--help` and `--print-schema` describe the tool itself, so they are
    // answered before a backend is required.
    for (auto arg : arguments) {
        if (arg == "--help" || arg == "-h") {
            print_usage(program);
            return 0;
        }
        if (arg == "--print-schema") {
            print_schema();
            return 0;
        }
    }
    Options options;
    auto error = luisa::string{};
    if (!parse_cli(arguments, options, error)) {
        if (!error.empty()) { report_failure(error); }
        print_usage(program);
        return 1;
    }
    Context context{program};
    if (!backend_is_installed(context, options.backend)) { return 1; }
    if (options.self_test) { return run_self_test(context, options); }
    // Build the effective document: either from a file, or from the embedded
    // default plus the command-line shaders.
    auto diagnostics = ns::Diagnostics{};
    auto document_dir = luisa::string{"."};
    auto document = ns::DispatchJson{};
    auto limits = ns::JsonLimits{};
    if (options.document.empty()) {
        if (!build_default_document(options.backend, options, document, error)) {
            report_failure(error);
            return 1;
        }
    } else {
        auto path = luisa::filesystem::path{};
        if (!luisa::path_from_narrow(options.document, path)) {
            report_failure(luisa::format("the document path '{}' is not usable",
                                         options.document));
            return 1;
        }
        document_dir = luisa::to_string(path.parent_path());
        if (document_dir.empty()) { document_dir = "."; }
        auto parsed = ns::parse_dispatch_file(path, limits);
        // The parse-stage diagnostics are carried into `diagnostics` instead of
        // being printed here: `strict` (from the document or from --strict, which
        // is applied below) turns every warning into an error, and a warning that
        // was already printed could no longer be escalated.
        for (auto &&diagnostic : parsed.errors) {
            diagnostics.errors.emplace_back(std::move(diagnostic));
        }
        for (auto &&warning : parsed.warnings) {
            diagnostics.warnings.emplace_back(std::move(warning));
        }
        if (!parsed.value.has_value()) {
            diagnostics.errors.emplace_back(luisa::format(
                "'{}' is not a usable dispatch document", options.document));
            report_diagnostics(diagnostics);
            return 1;
        }
        document = std::move(*parsed.value);
        merge_cli_shaders(options, document, diagnostics);
    }
    apply_cli_overrides(options, document);
    apply_log_level(document.config.log_level);
    // The codec cannot know `strict` while it parses, so the escalation happens
    // here: with --strict (or `config.strict`) every warning - parse stage,
    // command-line merging and semantic pass alike - is an error.
    if (document.config.strict) {
        for (auto &&warning : diagnostics.warnings) {
            diagnostics.errors.emplace_back(luisa::format("strict: {}", warning));
        }
        diagnostics.warnings.clear();
    }
    auto semantic_errors = luisa::vector<luisa::string>{};
    auto semantic_warnings = luisa::vector<luisa::string>{};
    auto semantics_ok = ns::validate_dispatch_semantics(document, document.config.limits,
                                                        semantic_errors, semantic_warnings);
    if (!semantics_ok && semantic_errors.empty()) {
        semantic_errors.emplace_back("the semantic validator rejected the document without a diagnostic");
    }
    for (auto &&diagnostic : semantic_errors) {
        diagnostics.errors.emplace_back(std::move(diagnostic));
    }
    for (auto &&diagnostic : semantic_warnings) {
        if (document.config.strict) {
            diagnostics.errors.emplace_back(luisa::format("strict: {}", diagnostic));
        } else {
            diagnostics.warnings.emplace_back(std::move(diagnostic));
        }
    }
    if (!diagnostics.errors.empty()) {
        report_diagnostics(diagnostics);
        return 1;
    }
    report_diagnostics(diagnostics);
    if (options.dump_dispatch) {
        auto written = ns::write_dispatch_json(document);
        if (!written.error.empty()) {
            report_failure(written.error);
            return 1;
        }
        auto path = luisa::filesystem::path{};
        if (!luisa::path_from_narrow(options.dump_path, path)) {
            report_failure(luisa::format("the dump path '{}' is not usable",
                                         options.dump_path));
            return 1;
        }
        if (auto write_error = luisa::string{}; !ns::write_output(
                path, luisa::span<const std::byte>{reinterpret_cast<const std::byte *>(written.json.data()), written.json.size()},
                "raw", uint3{0u, 0u, 0u}, PixelStorage::BYTE1, true, write_error)) {
            report_failure(write_error);
            return 1;
        }
        LUISA_INFO("wrote the effective dispatch document to '{}' ({} byte(s))",
                   options.dump_path, written.json.size());
        return 0;
    }
    auto backend = options.backend;
    if (!document.config.backend.empty() && document.config.backend != backend) {
        LUISA_WARNING("the document asks for backend '{}' but the command line "
                      "selected '{}': the command line wins",
                      document.config.backend, backend);
    }
    return run_document(context, backend, options, std::move(document), document_dir);
}
