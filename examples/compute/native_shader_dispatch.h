// Dispatch-document schema + JSON codec for the native shader example.
//
// This translation unit owns *data* only: the in-memory model of a dispatch
// document (see native_shader_examples/README.md), the parse/write API and the
// limits that bound both. It deliberately knows nothing about the device, the
// runtime or the filesystem layout of the sample data.
//
// Wire format (v1), key order as written by `write_dispatch_json`:
//
//   { "version": 1, "mode": {...}, "config": {...},
//     "shaders": [...], "resources": [...], "workflow": [...] }
//
// Every path is resolved by the caller against the document's directory.
// Every device handle is written as a resource *name*; a raw handle field is
// rejected (fail closed).
#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::native_shader {

// The schema names the runtime types it describes directly (a language, a pixel
// storage, ...); import them once instead of spelling the namespace everywhere.
using compute::NativeShaderLanguage;

// ---------------------------------------------------------------------------
// limits
// ---------------------------------------------------------------------------

// Every count/byte budget the parser and the semantic validator enforce. The
// caller passes them to `parse_dispatch_json`; `config.limits` may relax or
// tighten the *document* limits (resource/shader/command counts, inline and
// uniform payloads, and the byte size of a single resource) for the semantic
// pass that follows the parse.
struct JsonLimits {
    size_t max_document_bytes{64u << 20u};// input text size (before parsing)
    size_t max_resources{256u};
    size_t max_shaders{64u};
    size_t max_commands{4096u};
    size_t max_inline_bytes{32u << 20u};// decoded inline payload
    size_t max_uniform_bytes{65536u};   // per dispatch
    size_t max_bindings_per_dispatch{256u};
    // The byte size of one resource (a buffer's total, or the sum over all mip
    // levels of a texture/volume). Absurd values are refused here: a backend's
    // allocator aborts on a resource no device can hold, which would turn an
    // invalid document into a process crash instead of a diagnostic.
    size_t max_resource_bytes{16ull << 30u};
    size_t max_string_bytes{1u << 20u};
    size_t max_errors{32u}; // diagnostics collected per run
    uint32_t max_depth{64u};// JSON nesting depth
};

// ---------------------------------------------------------------------------
// enums
// ---------------------------------------------------------------------------

// The resource kinds the example supports. Curve BLAS (`curve`), motion-blur
// instances (`motion_instance`) and indirect dispatch buffers are *not* in this
// list: no backend implements curve or motion-blur acceleration structures,
// indirect dispatch is unsupported everywhere, and both are rejected with a
// dedicated diagnostic (see `unsupported_resource_types`).
enum class ResourceType : uint32_t {
    Buffer,
    Texture,
    Volume,
    BindlessArray,
    Accel,
    Mesh,
    ProceduralPrimitive,
};

// The command kinds the example executes. `curve_build` and
// `motion_instance_build` are not among them: no backend implements curve or
// motion-blur acceleration structures, so the codec rejects those names with a
// dedicated diagnostic (see `unsupported_command_kinds`). The tag mirrors the
// command classes of `luisa::compute`, plus two host-side pseudo-commands: `Log`
// and `Synchronize`.
enum class CommandKind : uint32_t {
    BufferUpload,
    BufferDownload,
    BufferCopy,
    TextureUpload,
    TextureDownload,
    TextureCopy,
    BufferToTextureCopy,
    TextureToBufferCopy,
    NativeDispatch,
    ShaderDispatch,
    BindlessArrayUpdate,
    MeshBuild,
    ProceduralPrimitiveBuild,
    AccelBuild,
    CustomCommand,
    Log,
    Synchronize,
};

// The number of `CommandKind` values, for tables indexed by the enumerator.
inline constexpr auto command_kind_count =
    luisa::to_underlying(CommandKind::Synchronize) + 1u;

[[nodiscard]] const char *to_string(ResourceType type) noexcept;
[[nodiscard]] const char *to_string(CommandKind kind) noexcept;
// The JSON spellings of the `cmd` discriminator.
[[nodiscard]] luisa::string_view command_kind_name(CommandKind kind) noexcept;
[[nodiscard]] bool parse_command_kind(luisa::string_view name, CommandKind &kind) noexcept;
[[nodiscard]] luisa::string_view resource_type_name(ResourceType type) noexcept;
[[nodiscard]] bool parse_resource_type(luisa::string_view name, ResourceType &type) noexcept;

// The command and resource names the example deliberately does not implement, so
// that a document asking for one gets a message saying *why* instead of a generic
// "unknown command". They are reported by `--print-schema` too.
[[nodiscard]] luisa::span<const luisa::string_view> command_kind_spellings() noexcept;
[[nodiscard]] luisa::span<const luisa::string_view> resource_type_spellings() noexcept;
[[nodiscard]] luisa::span<const luisa::string_view> unsupported_command_kinds() noexcept;
[[nodiscard]] luisa::span<const luisa::string_view> unsupported_resource_types() noexcept;

struct WindowJson {
    luisa::string title{"native shader"};
    uint32_t width{1024u};
    uint32_t height{1024u};
    bool vsync{true};
};

struct SnapshotJson {
    uint32_t every{0u};// 0 == off
    luisa::string path;
};

struct ModeJson {
    bool interactive{false};// "offline" (default) | "interactive"
    uint32_t frames{1u};    // offline: workflow executions
    bool gui{true};         // interactive only
    WindowJson window;
    luisa::string display_image;              // interactive: source texture resource
    luisa::string display_destination{"auto"};// "auto" | texture resource name
    float display_scale{1.0f};                // HDR exposure scale
    luisa::string display_kernel{"hdr_to_display"};
    bool dispatch_per_frame{true};
    uint32_t exit_after_frames{0u};// interactive: 0 == until closed
    SnapshotJson snapshot;
};

struct DStorageConfigJson {
    bool enabled{true};
    size_t staging_buffer_size{64u << 20u};
    luisa::string compression{"none"};// none | gdeflate
};

struct ConfigJson {
    luisa::string backend;                 // empty == take it from the CLI
    luisa::string default_language{"hlsl"};// hlsl | glsl | cuda_nvrtc
    uint32_t shader_model{65u};
    bool optimize{true};
    bool fast_math{false};
    bool debug_info{false};
    uint3 block_size{0u, 0u, 0u};// 0 => per-shader reflection
    bool has_push_constant_size{false};
    uint32_t push_constant_size{0u};// 0 => per-shader reflection
    luisa::vector<luisa::string> include_dirs;
    luisa::string output_dir{"native_shader_output"};
    DStorageConfigJson dstorage;
    bool strict{false};
    luisa::string log_level{"info"};
    JsonLimits limits;
};

// ---------------------------------------------------------------------------
// shaders
// ---------------------------------------------------------------------------

struct ShaderJson {
    luisa::string name;
    NativeShaderLanguage language{NativeShaderLanguage::HLSL};
    bool has_language{false};// false => derive from the file extension / config
    luisa::string path;      // "path": ...
    luisa::string source;    // "source": ...
    bool source_is_file{false};
    luisa::string entry_point;// empty => language default
    uint3 block_size{0u, 0u, 0u};
    uint32_t push_constant_size{0u};
    bool optimize{true};
    bool fast_math{false};
    bool debug_info{false};
    luisa::vector<luisa::string> include_dirs;
    bool from_cli{false};// merged in from the command line (override wins)
};

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

// Creation-time input of a resource: a file region read through dstorage (or a
// plain host read), an inline hex payload, or a region of another resource.
struct InputJson {
    enum class Kind : uint32_t { None,
                                 File,
                                 Inline,
                                 Resource };
    Kind kind{Kind::None};
    luisa::string file;
    size_t offset{0u};
    size_t size{0u};// 0 => rest of the file/resource
    luisa::string compression{"none"};
    luisa::vector<std::byte> inline_bytes;
    luisa::string resource;
};

struct ResourceJson {
    luisa::string name;
    ResourceType type{ResourceType::Buffer};
    // buffer
    luisa::string element;// float, float2..4, uint, uint2..4, int, int2..4, byte, triangle, aabb
    size_t count{0u};
    size_t byte_size{0u};// used when `element` is empty
    bool has_byte_size{false};
    // texture / volume
    luisa::string storage;// byte1..4, byte4_srgb, short1..4, int1..4, half1..4, ...
    uint3 size{0u, 0u, 0u};
    uint32_t levels{1u};
    // bindless array
    size_t slot_count{0u};
    luisa::string slot_type{"multiple"};// multiple | buffer | texture2d | texture3d
    // The ray-tracing resources this example creates. A mesh names its vertex and
    // triangle buffers, a procedural primitive its AABB buffer; the BLAS itself is
    // built by the matching `*_build` command. A `curve` or a `motion_instance` is
    // rejected - no backend implements them - see `unsupported_resource_types`.
    luisa::string vertex_buffer;
    luisa::string triangle_buffer;
    luisa::string aabb_buffer;
    InputJson input;
};

// ---------------------------------------------------------------------------
// workflow
// ---------------------------------------------------------------------------

struct BindingJson {
    uint32_t index{0u};
    bool has_index{false};
    uint32_t reg{0u};
    uint32_t space{0u};
    bool has_register{false};
    luisa::string resource;
    size_t offset{0u};
    size_t size{0u};            // 0 => rest of the resource
    luisa::string usage{"read"};// read | write | read_write
};

struct UniformJson {
    luisa::string type;// float32 | uint32 | int32 | float32x2..4 | uint32x2..4 | int32x2..4 | hex
    luisa::vector<std::byte> bytes;
    size_t alignment{4u};
};

struct ArgumentJson {
    luisa::string kind;// buffer | texture | bindless_array | accel | uniform
    luisa::string resource;
    size_t offset{0u};
    uint32_t level{0u};
    luisa::vector<std::byte> bytes;// uniform payload
    size_t alignment{4u};
};

struct SamplerJson {
    luisa::string filter{"point"};// point | linear_point | linear_linear | anisotropic
    luisa::string address{"edge"};// edge | repeat | mirror | zero
};

struct BindlessModJson {
    uint32_t slot{0u};
    luisa::string kind;         // buffer | texture2d | texture3d
    luisa::string op{"emplace"};// emplace | remove
    luisa::string resource;
    size_t offset{0u};
    size_t size{0u};// 0 => whole buffer
    SamplerJson sampler;
};

struct AccelModJson {
    uint32_t index{0u};
    bool has_user_id{false};
    uint32_t user_id{0u};
    bool has_visibility{false};
    uint32_t visibility{0u};
    bool has_opaque{false};
    bool opaque{true};
    bool has_transform{false};
    float transform[16]{};
    bool has_primitive{false};
    luisa::string primitive;
};

struct OutputJson {
    bool discard{true};
    luisa::string file;
    luisa::string format{"raw"};// raw | png
    bool overwrite{false};
};

// A host-side check run on a downloaded payload. `linear` compares
// `dst[i] == src[i] * k + c` against another resource; `copy` compares against
// another resource byte for byte; `none` disables the check.
struct VerifyJson {
    luisa::string kind{"none"};// none | linear | copy
    luisa::string source;
    float k{1.0f};
    float c{0.0f};
    float tolerance{0.0f};
};

struct CommandJson {
    CommandKind kind{CommandKind::Log};
    // resource names (never handles)
    luisa::string resource;
    luisa::string src;
    luisa::string dst;
    luisa::string buffer;
    luisa::string texture;
    luisa::string shader;
    luisa::string mode;   // bindless array update mode
    luisa::string request;// build request: prefer_update | force_build
    // byte/element offsets and sizes; every size of 0 means "rest"
    size_t offset{0u};
    size_t size{0u};
    size_t src_offset{0u};
    size_t dst_offset{0u};
    size_t buffer_offset{0u};
    uint32_t level{0u};
    uint32_t src_level{0u};
    uint32_t dst_level{0u};
    // The texture/volume region of `texture_*`, `buffer_to_texture_copy` and
    // `texture_to_buffer_copy`. In the document these are the `offset`/`size`/
    // `src_offset`/`dst_offset` keys written as 3-element arrays (a bare number
    // is the byte form, used by the buffer and the offset-only commands).
    uint3 offset3{0u, 0u, 0u};
    uint3 size3{0u, 0u, 0u};
    uint3 src_offset3{0u, 0u, 0u};
    uint3 dst_offset3{0u, 0u, 0u};
    luisa::string storage;
    InputJson input;
    OutputJson output;
    VerifyJson verify;
    // native dispatch
    uint3 dispatch{0u, 0u, 0u};// thread count
    uint3 grid{0u, 0u, 0u};    // thread-group count (alternative to dispatch)
    luisa::vector<BindingJson> bindings;
    luisa::vector<UniformJson> uniforms;
    bool allow_usage_override{false};
    // shader dispatch: `dispatch`/`grid`, or the `batched` list of thread counts.
    // There is no indirect form: no backend of this example supports one, so the
    // codec rejects the `indirect` key with a dedicated diagnostic.
    luisa::vector<ArgumentJson> arguments;
    luisa::vector<uint3> batched;
    // bindless array update
    luisa::vector<BindlessModJson> modifications;
    // mesh / procedural primitive / accel builds
    luisa::string vertex_buffer;
    luisa::string triangle_buffer;
    luisa::string aabb_buffer;
    size_t vertex_buffer_offset{0u};
    size_t vertex_buffer_size{0u};
    size_t triangle_buffer_offset{0u};
    size_t triangle_buffer_size{0u};
    size_t aabb_buffer_offset{0u};
    size_t aabb_buffer_size{0u};
    uint32_t vertex_stride{0u};
    uint32_t instance_count{0u};
    bool update_instance_buffer_only{false};
    luisa::vector<AccelModJson> accel_modifications;
    // custom command
    uint64_t uuid{0u};
    // pseudo-commands
    luisa::string message;
    luisa::string label;
};

// ---------------------------------------------------------------------------
// document
// ---------------------------------------------------------------------------

struct DispatchJson {
    uint32_t version{1u};
    ModeJson mode;
    ConfigJson config;
    luisa::vector<ShaderJson> shaders;
    luisa::vector<ResourceJson> resources;
    luisa::vector<CommandJson> workflow;
};

// ---------------------------------------------------------------------------
// hex helpers
// ---------------------------------------------------------------------------

// Strict lowercase/uppercase hex, even length, bounded by `max_bytes`.
[[nodiscard]] bool decode_hex(luisa::string_view text, size_t max_bytes,
                              luisa::vector<std::byte> &out,
                              luisa::string &error) noexcept;
[[nodiscard]] luisa::string encode_hex(luisa::span<const std::byte> bytes) noexcept;

// ---------------------------------------------------------------------------
// parse / write
// ---------------------------------------------------------------------------

struct ParseResult {
    luisa::optional<DispatchJson> value;
    luisa::vector<luisa::string> errors;// "<json path>: <message>"
    luisa::vector<luisa::string> warnings;
};

struct WriteOptions {
    bool pretty{true};
};

struct WriteResult {
    luisa::string json;
    luisa::string error;
};

// Structural parse: types, ranges, limits, unknown keys (warnings) and
// per-command field sets. No device and no filesystem access.
[[nodiscard]] ParseResult parse_dispatch_json(luisa::string_view text,
                                              const JsonLimits &limits) noexcept;
// Reads `path` (bounded by `limits.max_document_bytes`) and parses it.
[[nodiscard]] ParseResult parse_dispatch_file(const luisa::filesystem::path &path,
                                              const JsonLimits &limits) noexcept;

[[nodiscard]] WriteResult write_dispatch_json(const DispatchJson &document,
                                              const WriteOptions &options = {}) noexcept;

// The DSL kernels the example registers by name (see native_shader.cpp). A
// `shader_dispatch` may reference them without declaring a `shaders` entry; a
// `native_dispatch` may not, because it needs a native source. `--print-schema`
// reports the same list.
[[nodiscard]] luisa::span<const luisa::string_view> builtin_dsl_kernels() noexcept;

// The names the codec accepts for the two tables the runtime also has to map to
// device types (buffer elements and pixel storages). `--self-test` walks them,
// so the codec's validation table and the runtime's name -> enum table cannot
// drift apart.
[[nodiscard]] luisa::span<const luisa::string_view> buffer_element_spellings() noexcept;
[[nodiscard]] luisa::span<const luisa::string_view> pixel_storage_spellings() noexcept;
[[nodiscard]] luisa::span<const luisa::string_view> usage_spellings() noexcept;

// Semantic pass (no device): unique names, dangling references, kind/size
// agreement, per-command field consistency. `limits` is normally
// `document.config.limits`.
[[nodiscard]] bool validate_dispatch_semantics(const DispatchJson &document,
                                               const JsonLimits &limits,
                                               luisa::vector<luisa::string> &errors,
                                               luisa::vector<luisa::string> &warnings) noexcept;

// Canonical-form comparison used by the round-trip self test: two documents are
// equivalent when their canonical serialisations are identical.
[[nodiscard]] bool equivalent(const DispatchJson &a, const DispatchJson &b,
                              luisa::string &mismatch) noexcept;

}// namespace luisa::native_shader
