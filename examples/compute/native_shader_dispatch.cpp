// Dispatch-document JSON codec (yyjson). See native_shader_dispatch.h.
//
// Parsing is total and never raises: every problem becomes an entry of
// `ParseResult::errors` / `::warnings` (never a fatal log, never a trap),
// so that a single run reports as many document problems as it can.
// Serialisation mirrors the key order of the README tables, so that
// `write -> parse -> write` is byte-stable.

#include "native_shader_dispatch.h"

#include <yyjson.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <system_error>

#include <luisa/ast/usage.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::native_shader {

namespace {

// ---------------------------------------------------------------------------
// spelling tables
// ---------------------------------------------------------------------------
// Every enumerating key is matched case-insensitively with '-' and '_' treated
// as the same character; the canonical (lowercase, '_') spelling is what the
// document model stores and what the writer emits.

struct Spelling {
    const char *name;
    uint32_t value;
};

constexpr Spelling kCommandKindSpellings[]{
    {"buffer_upload", luisa::to_underlying(CommandKind::BufferUpload)},
    {"buffer_download", luisa::to_underlying(CommandKind::BufferDownload)},
    {"buffer_copy", luisa::to_underlying(CommandKind::BufferCopy)},
    {"texture_upload", luisa::to_underlying(CommandKind::TextureUpload)},
    {"texture_download", luisa::to_underlying(CommandKind::TextureDownload)},
    {"texture_copy", luisa::to_underlying(CommandKind::TextureCopy)},
    {"buffer_to_texture_copy", luisa::to_underlying(CommandKind::BufferToTextureCopy)},
    {"texture_to_buffer_copy", luisa::to_underlying(CommandKind::TextureToBufferCopy)},
    {"native_dispatch", luisa::to_underlying(CommandKind::NativeDispatch)},
    {"shader_dispatch", luisa::to_underlying(CommandKind::ShaderDispatch)},
    {"bindless_array_update", luisa::to_underlying(CommandKind::BindlessArrayUpdate)},
    {"mesh_build", luisa::to_underlying(CommandKind::MeshBuild)},
    {"procedural_primitive_build", luisa::to_underlying(CommandKind::ProceduralPrimitiveBuild)},
    {"accel_build", luisa::to_underlying(CommandKind::AccelBuild)},
    {"custom_command", luisa::to_underlying(CommandKind::CustomCommand)},
    {"log", luisa::to_underlying(CommandKind::Log)},
    {"synchronize", luisa::to_underlying(CommandKind::Synchronize)},
};

constexpr Spelling kResourceTypeSpellings[]{
    {"buffer", luisa::to_underlying(ResourceType::Buffer)},
    {"texture", luisa::to_underlying(ResourceType::Texture)},
    {"volume", luisa::to_underlying(ResourceType::Volume)},
    {"bindless_array", luisa::to_underlying(ResourceType::BindlessArray)},
    {"accel", luisa::to_underlying(ResourceType::Accel)},
    {"mesh", luisa::to_underlying(ResourceType::Mesh)},
    {"procedural_primitive", luisa::to_underlying(ResourceType::ProceduralPrimitive)},
};

constexpr Spelling kLanguageSpellings[]{
    {"hlsl", luisa::to_underlying(NativeShaderLanguage::HLSL)},
    {"glsl", luisa::to_underlying(NativeShaderLanguage::GLSL)},
    {"cuda_nvrtc", luisa::to_underlying(NativeShaderLanguage::CUDA_NVRTC)},
};

constexpr Spelling kBufferElementSpellings[]{
    {"float", 0u},
    {"float2", 0u},
    {"float3", 0u},
    {"float4", 0u},
    {"uint", 0u},
    {"uint2", 0u},
    {"uint3", 0u},
    {"uint4", 0u},
    {"int", 0u},
    {"int2", 0u},
    {"int3", 0u},
    {"int4", 0u},
    {"byte", 0u},
    // The ray-tracing payloads a `mesh` / `procedural_primitive` resource needs
    // (see `BufferElement` in native_shader_runtime.h).
    {"triangle", 0u},
    {"aabb", 0u},
};

constexpr Spelling kChannelElementSpellings[]{
    {"float", 0u},
    {"uint", 0u},
    {"int", 0u},
};

constexpr Spelling kStorageSpellings[]{
    {"byte1", 0u},
    {"byte2", 0u},
    {"byte4", 0u},
    {"byte4_srgb", 0u},
    {"short1", 0u},
    {"short2", 0u},
    {"short4", 0u},
    {"int1", 0u},
    {"int2", 0u},
    {"int4", 0u},
    {"half1", 0u},
    {"half2", 0u},
    {"half4", 0u},
    {"float1", 0u},
    {"float2", 0u},
    {"float4", 0u},
    {"r10g10b10a2", 0u},
    {"r11g11b10", 0u},
};

constexpr Spelling kUsageSpellings[]{
    {"none", luisa::to_underlying(compute::Usage::NONE)},
    {"read", luisa::to_underlying(compute::Usage::READ)},
    {"write", luisa::to_underlying(compute::Usage::WRITE)},
    {"read_write", luisa::to_underlying(compute::Usage::READ_WRITE)},
};

constexpr Spelling kCompressionSpellings[]{
    {"none", 0u},
    {"gdeflate", 0u},
};

constexpr Spelling kSlotTypeSpellings[]{
    {"multiple", 0u},
    {"buffer", 1u},
    {"texture2d", 2u},
    {"texture3d", 3u},
};

constexpr Spelling kBindlessKindSpellings[]{
    {"buffer", 1u},
    {"texture2d", 2u},
    {"texture3d", 3u},
};

constexpr Spelling kBindlessOpSpellings[]{
    {"emplace", 0u},
    {"remove", 1u},
};

constexpr Spelling kRequestSpellings[]{
    {"prefer_update", 0u},
    {"force_build", 1u},
};

constexpr Spelling kFilterSpellings[]{
    {"point", 0u},
    {"linear_point", 1u},
    {"linear_linear", 2u},
    {"anisotropic", 3u},
};

constexpr Spelling kAddressSpellings[]{
    {"edge", 0u},
    {"repeat", 1u},
    {"mirror", 2u},
    {"zero", 3u},
};

constexpr Spelling kOutputFormatSpellings[]{
    {"raw", 0u},
    {"png", 1u},
};

constexpr Spelling kLogLevelSpellings[]{
    {"verbose", 0u},
    {"info", 1u},
    {"warning", 2u},
    {"error", 3u},
};

constexpr Spelling kVerifyKindSpellings[]{
    {"none", 0u},
    {"linear", 1u},
    {"copy", 2u},
};

constexpr Spelling kArgumentKindSpellings[]{
    {"buffer", 0u},
    {"texture", 1u},
    {"bindless_array", 2u},
    {"accel", 3u},
    {"uniform", 4u},
};

constexpr Spelling kSourceTypeSpellings[]{
    {"file", 1u},
    {"code", 0u},
};

constexpr Spelling kModeTypeSpellings[]{
    {"offline", 0u},
    {"interactive", 1u},
};

constexpr Spelling kUuidSpellings[]{
    {"native_shader_dispatch",
     luisa::to_underlying(compute::CustomCommandUUID::NATIVE_SHADER_DISPATCH)},
    {"dstorage_read",
     luisa::to_underlying(compute::CustomCommandUUID::DSTORAGE_READ)},
};

// The uniform payload spellings of the native-shader launcher.
constexpr Spelling kUniformTypeSpellings[]{
    {"float32", 0u},
    {"float32x2", 0u},
    {"float32x3", 0u},
    {"float32x4", 0u},
    {"uint32", 0u},
    {"uint32x2", 0u},
    {"uint32x3", 0u},
    {"uint32x4", 0u},
    {"int32", 0u},
    {"int32x2", 0u},
    {"int32x3", 0u},
    {"int32x4", 0u},
    {"hex", 0u},
};

[[nodiscard]] luisa::string canonical_spelling(luisa::string_view text) noexcept {
    auto result = luisa::string{text.data(), text.size()};
    for (auto &c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        } else if (c == '-') {
            c = '_';
        }
    }
    return result;
}

[[nodiscard]] luisa::string_view view_of(const luisa::string &text) noexcept {
    return luisa::string_view{text.data(), text.size()};
}

template<size_t N>
[[nodiscard]] const Spelling *find_spelling(const Spelling (&table)[N],
                                            luisa::string_view canonical) noexcept {
    for (auto i = 0u; i < N; i++) {
        if (canonical == luisa::string_view{table[i].name}) { return &table[i]; }
    }
    return nullptr;
}

template<size_t N>
[[nodiscard]] luisa::string spelling_list(const Spelling (&table)[N]) noexcept {
    auto result = luisa::string{};
    for (auto i = 0u; i < N; i++) {
        if (i != 0u) { result.append(", "); }
        result.append(table[i].name);
    }
    return result;
}

// ---------------------------------------------------------------------------
// object key sets
// ---------------------------------------------------------------------------

struct KeySet {
    const luisa::string_view *keys{nullptr};
    size_t count{0u};
};

template<size_t N>
[[nodiscard]] constexpr KeySet keys_of(const luisa::string_view (&keys)[N]) noexcept {
    return KeySet{keys, N};
}

[[nodiscard]] bool contains_key(const KeySet &set, luisa::string_view key) noexcept {
    for (auto i = size_t{0u}; i < set.count; i++) {
        if (set.keys[i] == key) { return true; }
    }
    return false;
}

// The keys of `set`, comma separated: a diagnostic that rejects a key names the
// set that the command does accept.
[[nodiscard]] luisa::string key_list(const KeySet &set) noexcept {
    auto result = luisa::string{};
    for (auto i = size_t{0u}; i < set.count; i++) {
        if (i != 0u) { result.append(", "); }
        result.append(set.keys[i].data(), set.keys[i].size());
    }
    return result;
}

// ---------------------------------------------------------------------------
// per-command key sets
// ---------------------------------------------------------------------------
// The keys each `cmd` accepts, in the order of the README table (which is also
// the order `write_dispatch_json` emits them in).  A key that is present but
// not allowed for the command's kind is an error when some other command kind
// owns it - a genuine mistake - and a warning otherwise (a typo or a key that
// this version does not know yet).

constexpr luisa::string_view kBufferUploadKeys[]{"resource", "offset", "size", "input"};
constexpr luisa::string_view kBufferDownloadKeys[]{"resource", "offset", "size", "output", "verify"};
constexpr luisa::string_view kBufferCopyKeys[]{"src", "src_offset", "dst", "dst_offset", "size"};
constexpr luisa::string_view kTextureUploadKeys[]{"resource", "level", "offset", "size", "storage", "input"};
constexpr luisa::string_view kTextureDownloadKeys[]{"resource", "level", "offset", "size", "storage", "output"};
constexpr luisa::string_view kTextureCopyKeys[]{"storage", "src", "dst", "src_level", "dst_level", "size", "src_offset", "dst_offset"};
constexpr luisa::string_view kBufferToTextureCopyKeys[]{"buffer", "buffer_offset", "texture", "storage", "level", "size", "offset", "texture_offset"};
constexpr luisa::string_view kTextureToBufferCopyKeys[]{"buffer", "buffer_offset", "texture", "storage", "level", "size", "offset", "texture_offset"};
constexpr luisa::string_view kNativeDispatchKeys[]{"shader", "dispatch", "grid", "bindings", "uniforms", "allow_usage_override"};
constexpr luisa::string_view kShaderDispatchKeys[]{"shader", "arguments", "dispatch", "indirect", "batched"};
constexpr luisa::string_view kBindlessArrayUpdateKeys[]{"resource", "mode", "modifications"};
constexpr luisa::string_view kMeshBuildKeys[]{
    "resource", "request", "vertex_buffer", "vertex_buffer_offset", "vertex_buffer_size",
    "vertex_stride", "triangle_buffer", "triangle_buffer_offset", "triangle_buffer_size"};
constexpr luisa::string_view kProceduralPrimitiveBuildKeys[]{
    "resource", "request", "aabb_buffer", "aabb_buffer_offset", "aabb_buffer_size"};
constexpr luisa::string_view kAccelBuildKeys[]{
    "resource", "instance_count", "request", "update_instance_buffer_only", "modifications"};
// The type-specific fields of the two registered custom commands. The
// `native_shader_dispatch` alias accepts the whole native-dispatch field set
// (`native_dispatch` in the alias form), the `dstorage_read` command reads an
// `input` region into `resource` at `offset`. Unknown uuids are rejected by the
// semantic pass.
constexpr luisa::string_view kCustomCommandKeys[]{
    "uuid", "resource", "buffer", "buffer_offset", "texture", "storage", "level",
    "offset", "size", "input", "output", "label",
    "shader", "dispatch", "grid", "bindings", "uniforms", "allow_usage_override"};
constexpr luisa::string_view kLogKeys[]{"message"};
constexpr luisa::string_view kSynchronizeKeys[]{"label"};

// The union of the sets above: a present key inside it is "meaningful for a
// different cmd kind" and therefore an error for the current command. `indirect`
// is kept here (and in `kShaderDispatchKeys`) although no backend implements it,
// so that a `shader_dispatch` carrying one is recognised and rejected by its own
// diagnostic instead of being taken for a typo.
constexpr luisa::string_view kAllCommandKeys[]{
    "resource", "src", "src_offset", "dst", "dst_offset", "size", "offset", "buffer",
    "buffer_offset", "texture", "storage", "level", "src_level", "dst_level", "input",
    "output", "verify", "shader", "dispatch", "grid", "bindings", "uniforms",
    "allow_usage_override", "arguments", "indirect", "batched", "mode", "modifications",
    "request", "vertex_buffer", "vertex_buffer_offset", "vertex_buffer_size", "vertex_stride",
    "triangle_buffer", "triangle_buffer_offset", "triangle_buffer_size", "aabb_buffer",
    "aabb_buffer_offset", "aabb_buffer_size", "instance_count",
    "update_instance_buffer_only", "uuid", "message", "label"};

[[nodiscard]] KeySet command_key_set(CommandKind kind) noexcept {
    switch (kind) {
        case CommandKind::BufferUpload: return keys_of(kBufferUploadKeys);
        case CommandKind::BufferDownload: return keys_of(kBufferDownloadKeys);
        case CommandKind::BufferCopy: return keys_of(kBufferCopyKeys);
        case CommandKind::TextureUpload: return keys_of(kTextureUploadKeys);
        case CommandKind::TextureDownload: return keys_of(kTextureDownloadKeys);
        case CommandKind::TextureCopy: return keys_of(kTextureCopyKeys);
        case CommandKind::BufferToTextureCopy: return keys_of(kBufferToTextureCopyKeys);
        case CommandKind::TextureToBufferCopy: return keys_of(kTextureToBufferCopyKeys);
        case CommandKind::NativeDispatch: return keys_of(kNativeDispatchKeys);
        case CommandKind::ShaderDispatch: return keys_of(kShaderDispatchKeys);
        case CommandKind::BindlessArrayUpdate: return keys_of(kBindlessArrayUpdateKeys);
        case CommandKind::MeshBuild: return keys_of(kMeshBuildKeys);
        case CommandKind::ProceduralPrimitiveBuild: return keys_of(kProceduralPrimitiveBuildKeys);
        case CommandKind::AccelBuild: return keys_of(kAccelBuildKeys);
        case CommandKind::CustomCommand: return keys_of(kCustomCommandKeys);
        case CommandKind::Log: return keys_of(kLogKeys);
        case CommandKind::Synchronize: return keys_of(kSynchronizeKeys);
    }
    return KeySet{};
}

// The keys every other object of the document accepts.
constexpr luisa::string_view kRootKeys[]{"version", "mode", "config", "shaders", "resources", "workflow"};
constexpr luisa::string_view kModeKeys[]{
    "type", "frames", "gui", "window", "display_image", "display_destination", "display_scale",
    "display_kernel", "dispatch_per_frame", "exit_after_frames", "snapshot"};
constexpr luisa::string_view kWindowKeys[]{"title", "width", "height", "vsync"};
constexpr luisa::string_view kSnapshotKeys[]{"every", "path"};
constexpr luisa::string_view kConfigKeys[]{
    "backend", "default_language", "shader_model", "optimize", "fast_math", "debug_info",
    "block_size", "push_constant_size", "include_dirs", "output_dir", "dstorage", "strict",
    "log_level", "limits"};
constexpr luisa::string_view kDStorageKeys[]{"enabled", "staging_buffer_size", "compression"};
constexpr luisa::string_view kLimitKeys[]{
    "max_document_bytes", "max_resources", "max_shaders", "max_commands", "max_inline_bytes",
    "max_uniform_bytes", "max_bindings_per_dispatch", "max_resource_bytes", "max_string_bytes",
    "max_errors", "max_depth"};
constexpr luisa::string_view kShaderKeys[]{
    "name", "language", "path", "source", "source_type", "entry_point", "block_size",
    "push_constant_size", "include_dirs", "optimize", "fast_math", "debug_info"};
constexpr luisa::string_view kCommonResourceKeys[]{"name", "type", "input"};
constexpr luisa::string_view kBufferResourceKeys[]{"name", "type", "element", "count", "byte_size", "input"};
constexpr luisa::string_view kTextureResourceKeys[]{"name", "type", "storage", "size", "levels", "element", "input"};
constexpr luisa::string_view kVolumeResourceKeys[]{"name", "type", "storage", "size", "levels", "element", "input"};
constexpr luisa::string_view kBindlessResourceKeys[]{"name", "type", "slot_count", "slot_type", "input"};
constexpr luisa::string_view kMeshResourceKeys[]{"name", "type", "vertex_buffer", "triangle_buffer", "input"};
// A procedural primitive names the buffer of its creation-time AABB range. That
// range is optional - the matching `procedural_primitive_build` command carries
// the buffer its BLAS is built from - but when it is named the buffer must hold
// `aabb` elements (see `validate_resource_entry`).
constexpr luisa::string_view kProceduralPrimitiveResourceKeys[]{"name", "type", "aabb_buffer", "input"};
constexpr luisa::string_view kInputKeys[]{"file", "inline", "resource", "offset", "size", "compression"};
constexpr luisa::string_view kInlineKeys[]{"hex"};
constexpr luisa::string_view kOutputKeys[]{"discard", "file", "format", "overwrite"};
constexpr luisa::string_view kVerifyKeys[]{"kind", "source", "k", "c", "tolerance"};
constexpr luisa::string_view kBindingKeys[]{"index", "register", "space", "resource", "offset", "size", "usage"};
constexpr luisa::string_view kUniformKeys[]{"type", "value", "hex"};
constexpr luisa::string_view kArgumentKeys[]{"kind", "resource", "offset", "level", "type", "value", "hex"};
constexpr luisa::string_view kModificationKeys[]{"slot", "kind", "op", "resource", "offset", "size", "sampler"};
constexpr luisa::string_view kSamplerKeys[]{"filter", "address"};
constexpr luisa::string_view kAccelModificationKeys[]{
    "index", "user_id", "visibility", "opaque", "transform", "primitive"};

// ---------------------------------------------------------------------------
// element / storage sizes
// ---------------------------------------------------------------------------

// `float3`, `uint3`, ... : the scalar kind and the component count of a buffer
// element type.  Returns false for an unknown spelling.
[[nodiscard]] bool buffer_element_layout(luisa::string_view canonical,
                                         uint32_t &scalar_bytes,
                                         uint32_t &components) noexcept {
    struct ElementLayout {
        const char *name;
        uint32_t scalar_bytes;
        uint32_t components;
    };
    constexpr ElementLayout kLayouts[]{
        {"byte", 1u, 1u},
        {"half", 2u, 1u},
        {"half2", 2u, 2u},
        {"half3", 2u, 3u},
        {"half4", 2u, 4u},
        {"float", 4u, 1u},
        {"float2", 4u, 2u},
        {"float3", 4u, 3u},
        {"float4", 4u, 4u},
        {"uint", 4u, 1u},
        {"uint2", 4u, 2u},
        {"uint3", 4u, 3u},
        {"uint4", 4u, 4u},
        {"int", 4u, 1u},
        {"int2", 4u, 2u},
        {"int3", 4u, 3u},
        {"int4", 4u, 4u},
        // The two ray-tracing payloads: a `Triangle` is three u32 vertex indices
        // (12 bytes) and an `AABB` is six floats (24 bytes).
        {"triangle", 4u, 3u},
        {"aabb", 4u, 6u},
    };
    for (auto layout : kLayouts) {
        if (canonical == luisa::string_view{layout.name}) {
            scalar_bytes = layout.scalar_bytes;
            components = layout.components;
            return true;
        }
    }
    return false;
}

[[nodiscard]] size_t buffer_element_bytes(luisa::string_view canonical) noexcept {
    auto scalar_bytes = 0u;
    auto components = 0u;
    return buffer_element_layout(canonical, scalar_bytes, components) ?
               static_cast<size_t>(scalar_bytes) * components :
               0u;
}

// Bytes occupied by one texel/volume element of a pixel storage;
// `pixel_storage_size` reports the same unit size.
[[nodiscard]] size_t storage_bytes(luisa::string_view canonical) noexcept {
    if (canonical == "byte1") { return 1u; }
    if (canonical == "byte2") { return 2u; }
    if (canonical == "byte4" || canonical == "byte4_srgb") { return 4u; }
    if (canonical == "short1" || canonical == "half1") { return 2u; }
    if (canonical == "short2" || canonical == "half2") { return 4u; }
    if (canonical == "short4" || canonical == "half4") { return 8u; }
    if (canonical == "int1" || canonical == "float1" ||
        canonical == "r10g10b10a2" || canonical == "r11g11b10") { return 4u; }
    if (canonical == "int2" || canonical == "float2") { return 8u; }
    if (canonical == "int4" || canonical == "float4") { return 16u; }
    return 0u;
}

// The scalar channel type implied by a storage: `float*` reads as float, `int*`
// as int, everything else as uint (see the README).
[[nodiscard]] const char *default_channel_element(luisa::string_view storage) noexcept {
    if (storage.starts_with("float")) { return "float"; }
    if (storage.starts_with("int")) { return "int"; }
    return "uint";
}

// The extent of mip level `level`: `max(1, extent >> level)` per component.
[[nodiscard]] uint3 mip_extent(uint3 size, uint32_t level) noexcept {
    auto shift = [level](uint32_t extent) noexcept {
        auto shifted = level >= 32u ? 0u : extent >> level;
        return shifted == 0u ? 1u : shifted;
    };
    return uint3{shift(size.x), shift(size.y), shift(size.z)};
}

// ---------------------------------------------------------------------------
// hex
// ---------------------------------------------------------------------------

[[nodiscard]] int hex_nibble(char c) noexcept {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

class Diagnostics {

public:
    explicit Diagnostics(const JsonLimits &limits) noexcept
        : _max_errors{limits.max_errors} {}

    // The semantic pass is re-run over the diagnostics of the parse pass, so a
    // warning is never reported twice.
    Diagnostics(const JsonLimits &limits,
                const luisa::vector<luisa::string> &known_warnings) noexcept
        : _max_errors{limits.max_errors} {
        for (auto &warning : known_warnings) { _seen_warnings.emplace(warning); }
    }

    // `max_errors` bounds the vector so that a hostile document cannot grow it
    // without bound.  Parsing continues either way; `has_error` stays true even
    // when nothing could be recorded, so the caller still fails closed.
    void error(luisa::string message) noexcept {
        _has_error = true;
        if (_errors.size() < _max_errors) { _errors.emplace_back(std::move(message)); }
    }

    void warning(luisa::string message) noexcept {
        if (_seen_warnings.emplace(message).second) { _warnings.emplace_back(std::move(message)); }
    }

    [[nodiscard]] bool has_error() const noexcept { return _has_error; }
    [[nodiscard]] const luisa::vector<luisa::string> &errors() const noexcept { return _errors; }
    [[nodiscard]] const luisa::vector<luisa::string> &warnings() const noexcept { return _warnings; }

private:
    size_t _max_errors{32u};
    bool _has_error{false};
    luisa::vector<luisa::string> _errors;
    luisa::vector<luisa::string> _warnings;
    luisa::unordered_set<luisa::string> _seen_warnings;
};

// ---------------------------------------------------------------------------
// json pointers
// ---------------------------------------------------------------------------
// Diagnostics are addressed like `workflow[3].dispatch` or `config.strict`.

[[nodiscard]] luisa::string path_key(luisa::string_view base, luisa::string_view key) noexcept {
    auto result = luisa::string{base.data(), base.size()};
    if (!result.empty()) { result.push_back('.'); }
    result.append(key.data(), key.size());
    return result;
}

[[nodiscard]] luisa::string path_index(luisa::string_view base, size_t index) noexcept {
    return luisa::format(FMT_STRING("{}[{}]"), base, index);
}

[[nodiscard]] const char *json_type_name(const yyjson_val *value) noexcept {
    if (value == nullptr) { return "missing"; }
    if (yyjson_is_null(value)) { return "null"; }
    if (yyjson_is_bool(value)) { return "bool"; }
    if (yyjson_is_real(value)) { return "real"; }
    if (yyjson_is_num(value)) { return "int"; }
    if (yyjson_is_str(value)) { return "string"; }
    if (yyjson_is_arr(value)) { return "array"; }
    if (yyjson_is_obj(value)) { return "object"; }
    return "unknown";
}

// ---------------------------------------------------------------------------
// parse context
// ---------------------------------------------------------------------------

struct Ctx {
    Diagnostics &diag;
    const JsonLimits &limits;
    uint32_t depth{0u};
};

// A recursive object/array converter; the JSON reader itself has no depth
// bound, so `max_depth` is enforced here.
struct DepthGuard {
    explicit DepthGuard(Ctx &ctx) noexcept : _ctx{ctx} { _ctx.depth++; }
    DepthGuard(const DepthGuard &) = delete;
    DepthGuard &operator=(const DepthGuard &) = delete;
    ~DepthGuard() noexcept { _ctx.depth--; }

private:
    Ctx &_ctx;
};

[[nodiscard]] bool check_depth(Ctx &ctx, luisa::string_view path) noexcept {
    if (ctx.depth >= ctx.limits.max_depth) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: nesting depth exceeds the limit of {}"),
                                     path, ctx.limits.max_depth));
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// type-checked accessors
// ---------------------------------------------------------------------------
// Every accessor reports the exact path and the JSON type of the offending
// value, and leaves the destination untouched on failure.

[[nodiscard]] luisa::string expected_message(luisa::string_view path,
                                             const char *expected,
                                             const yyjson_val *value) noexcept {
    return luisa::format(FMT_STRING("{}: expected {}, got {}"),
                         path, expected, json_type_name(value));
}

[[nodiscard]] const yyjson_val *expect_object(Ctx &ctx, const yyjson_val *value,
                                              luisa::string_view path) noexcept {
    if (value == nullptr || !yyjson_is_obj(value)) {
        ctx.diag.error(expected_message(path, "object", value));
        return nullptr;
    }
    return value;
}

[[nodiscard]] const yyjson_val *expect_array(Ctx &ctx, const yyjson_val *value,
                                             luisa::string_view path) noexcept {
    if (value == nullptr || !yyjson_is_arr(value)) {
        ctx.diag.error(expected_message(path, "array", value));
        return nullptr;
    }
    return value;
}

[[nodiscard]] luisa::optional<bool> expect_bool(Ctx &ctx, const yyjson_val *value,
                                                luisa::string_view path) noexcept {
    if (value == nullptr || !yyjson_is_bool(value)) {
        ctx.diag.error(expected_message(path, "bool", value));
        return luisa::nullopt;
    }
    return yyjson_get_bool(value);
}

[[nodiscard]] luisa::optional<luisa::string> expect_string(Ctx &ctx, const yyjson_val *value,
                                                           luisa::string_view path) noexcept {
    if (value == nullptr || !yyjson_is_str(value)) {
        ctx.diag.error(expected_message(path, "string", value));
        return luisa::nullopt;
    }
    auto length = yyjson_get_len(value);
    if (length > ctx.limits.max_string_bytes) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: string of {} bytes exceeds the limit of {} bytes"),
                                     path, length, ctx.limits.max_string_bytes));
        return luisa::nullopt;
    }
    return luisa::string{yyjson_get_str(value), length};
}

// Signed integers are checked first: yyjson stores a positive integer as an
// unsigned value and yyjson_get_sint() wraps for anything above INT64_MAX.
[[nodiscard]] luisa::optional<int64_t> expect_i64(Ctx &ctx, const yyjson_val *value,
                                                  luisa::string_view path) noexcept {
    if (value == nullptr) {
        ctx.diag.error(expected_message(path, "integer", value));
        return luisa::nullopt;
    }
    if (yyjson_is_sint(value)) { return yyjson_get_sint(value); }
    if (yyjson_is_uint(value)) {
        auto raw = yyjson_get_uint(value);
        if (raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: value {} is out of range for a 64-bit integer"),
                                         path, raw));
            return luisa::nullopt;
        }
        return static_cast<int64_t>(raw);
    }
    if (yyjson_is_real(value)) {
        auto real = yyjson_get_real(value);
        if (!std::isfinite(real)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected a finite integer, got {}"), path, real));
            return luisa::nullopt;
        }
        if (std::floor(real) != real) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected an integer, got the real {}"), path, real));
            return luisa::nullopt;
        }
        if (real < -9223372036854775808.0 || real >= 9223372036854775808.0) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: value {} is out of range for a 64-bit integer"),
                                         path, real));
            return luisa::nullopt;
        }
        return static_cast<int64_t>(real);
    }
    ctx.diag.error(expected_message(path, "integer", value));
    return luisa::nullopt;
}

// A non-negative integer: negative values, fractional values and values above
// UINT64_MAX are rejected.  An integral real (`1.0`) is accepted.
[[nodiscard]] luisa::optional<uint64_t> expect_u64(Ctx &ctx, const yyjson_val *value,
                                                   luisa::string_view path) noexcept {
    if (value == nullptr) {
        ctx.diag.error(expected_message(path, "non-negative integer", value));
        return luisa::nullopt;
    }
    if (yyjson_is_uint(value)) { return yyjson_get_uint(value); }
    if (yyjson_is_sint(value)) {
        auto signed_value = expect_i64(ctx, value, path);
        if (!signed_value) { return luisa::nullopt; }
        if (*signed_value < 0) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected a non-negative integer, got {}"),
                                         path, *signed_value));
            return luisa::nullopt;
        }
        return static_cast<uint64_t>(*signed_value);
    }
    if (yyjson_is_real(value)) {
        auto real = yyjson_get_real(value);
        if (!std::isfinite(real)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected a finite integer, got {}"), path, real));
            return luisa::nullopt;
        }
        if (std::floor(real) != real) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected an integer, got the real {}"), path, real));
            return luisa::nullopt;
        }
        if (real < 0.0) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected a non-negative integer, got {}"),
                                         path, real));
            return luisa::nullopt;
        }
        if (real >= 18446744073709551616.0) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: value {} is out of range for a 64-bit integer"),
                                         path, real));
            return luisa::nullopt;
        }
        return static_cast<uint64_t>(real);
    }
    ctx.diag.error(expected_message(path, "non-negative integer", value));
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<uint32_t> expect_u32(Ctx &ctx, const yyjson_val *value,
                                                   luisa::string_view path) noexcept {
    auto raw = expect_u64(ctx, value, path);
    if (!raw) { return luisa::nullopt; }
    if (*raw > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: value {} exceeds UINT32_MAX"),
                                     path, *raw));
        return luisa::nullopt;
    }
    return static_cast<uint32_t>(*raw);
}

// `uint64_t` values fit into an unsigned type of the same width; the comparison
// is only meaningful for a narrower target (a template so that the always-true
// case is discarded instead of warned about).
template<typename Unsigned>
[[nodiscard]] bool fits_in(uint64_t value) noexcept {
    if constexpr (sizeof(Unsigned) >= sizeof(uint64_t)) {
        (void)value;
        return true;
    } else {
        return value <= static_cast<uint64_t>(std::numeric_limits<Unsigned>::max());
    }
}

[[nodiscard]] luisa::optional<size_t> expect_size(Ctx &ctx, const yyjson_val *value,
                                                  luisa::string_view path) noexcept {
    auto raw = expect_u64(ctx, value, path);
    if (!raw) { return luisa::nullopt; }
    if (!fits_in<size_t>(*raw)) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: value {} exceeds the addressable size"),
                                     path, *raw));
        return luisa::nullopt;
    }
    return static_cast<size_t>(*raw);
}

[[nodiscard]] luisa::optional<double> expect_f64(Ctx &ctx, const yyjson_val *value,
                                                 luisa::string_view path) noexcept {
    if (value == nullptr) {
        ctx.diag.error(expected_message(path, "number", value));
        return luisa::nullopt;
    }
    if (yyjson_is_real(value)) {
        auto real = yyjson_get_real(value);
        if (!std::isfinite(real)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: expected a finite number, got {}"), path, real));
            return luisa::nullopt;
        }
        return real;
    }
    if (yyjson_is_uint(value)) { return static_cast<double>(yyjson_get_uint(value)); }
    if (yyjson_is_sint(value)) { return static_cast<double>(yyjson_get_sint(value)); }
    ctx.diag.error(expected_message(path, "number", value));
    return luisa::nullopt;
}

[[nodiscard]] luisa::optional<float> expect_f32(Ctx &ctx, const yyjson_val *value,
                                                luisa::string_view path) noexcept {
    auto real = expect_f64(ctx, value, path);
    if (!real) { return luisa::nullopt; }
    auto result = static_cast<float>(*real);
    if (!std::isfinite(result)) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: value {} is out of range for a 32-bit float"),
                                     path, *real));
        return luisa::nullopt;
    }
    return result;
}

// Exactly `N` (or, for the two-element texture/volume form, `2..3`) array
// elements of `T`; every other shape - including a shorter array, a string or
// an object - is an error.
template<typename T, size_t N>
    requires std::is_same_v<T, float> || std::is_same_v<T, uint32_t>
[[nodiscard]] bool expect_pod_array(Ctx &ctx, const yyjson_val *value, luisa::string_view path,
                                    T (&out)[N], const char *element_name) noexcept {
    if (value == nullptr || !yyjson_is_arr(value)) {
        auto expected = luisa::format(FMT_STRING("{}-element array of {}"), N, element_name);
        ctx.diag.error(expected_message(path, expected.c_str(), value));
        return false;
    }
    if (yyjson_arr_size(value) != N) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: expected a {}-element array of {}, got {} elements"),
                                     path, N, element_name, yyjson_arr_size(value)));
        return false;
    }
    for (auto i = size_t{0u}; i < N; i++) {
        auto element_path = path_index(path, i);
        auto *element = yyjson_arr_get(value, i);
        if constexpr (std::is_same_v<T, float>) {
            auto parsed = expect_f32(ctx, element, element_path);
            if (!parsed) { return false; }
            out[i] = *parsed;
        } else {
            auto parsed = expect_u32(ctx, element, element_path);
            if (!parsed) { return false; }
            out[i] = *parsed;
        }
    }
    return true;
}

[[nodiscard]] luisa::optional<uint3> expect_uint3(Ctx &ctx, const yyjson_val *value,
                                                  luisa::string_view path,
                                                  bool allow_two_components) noexcept {
    // The texture/volume form of `size` accepts `[w, h]` as well as `[w, h, d]`,
    // and the message has to say so.
    auto expected = allow_two_components ? "a 2- or 3-element array of u32" : "a 3-element array of u32";
    if (value == nullptr || !yyjson_is_arr(value)) {
        ctx.diag.error(expected_message(path, expected, value));
        return luisa::nullopt;
    }
    auto count = yyjson_arr_size(value);
    if (count != 3u && !(allow_two_components && count == 2u)) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: expected {}, got {} element(s)"),
                                     path, expected, count));
        return luisa::nullopt;
    }
    auto result = uint3{0u, 0u, 1u};
    uint32_t *components[3]{&result.x, &result.y, &result.z};
    for (auto i = size_t{0u}; i < count; i++) {
        auto parsed = expect_u32(ctx, yyjson_arr_get(value, i), path_index(path, i));
        if (!parsed) { return luisa::nullopt; }
        *components[i] = *parsed;
    }
    return result;
}

// ---------------------------------------------------------------------------
// unknown / duplicate keys
// ---------------------------------------------------------------------------

// Walks every key of `object` and reports
//   * duplicate keys (a warning: the *first* occurrence wins, exactly like
//     yyjson_obj_get(), which is what every reader below uses),
//   * a `handle` key, which the wire format forbids (fail closed),
//   * keys that belong to a *different* `cmd` (an error, `cmd` non-empty),
//   * every other key (a warning; `--strict` escalates warnings into errors,
//     which the orchestrator does by re-running the validation).
void check_object_keys(Ctx &ctx, const yyjson_val *object, luisa::string_view path,
                       const KeySet &allowed, luisa::string_view cmd) noexcept {
    auto seen = luisa::unordered_set<luisa::string_view>{};
    auto iter = yyjson_obj_iter_with(object);
    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
        auto *name = yyjson_get_str(key);
        auto length = yyjson_get_len(key);
        auto view = luisa::string_view{name, length};
        if (!seen.emplace(view).second) {
            ctx.diag.warning(luisa::format(FMT_STRING("{}: duplicate key"), path_key(path, view)));
            continue;
        }
        if (contains_key(allowed, view)) { continue; }
        // `cmd` is the discriminator of a command object, so it is not part of
        // the per-kind key set it selects.
        if (!cmd.empty() && view == "cmd") { continue; }
        if (view == "handle") {
            ctx.diag.error(luisa::format(FMT_STRING("{}: the document never carries device handles; name a resource instead"),
                                         path_key(path, view)));
        } else if (!cmd.empty() && contains_key(keys_of(kAllCommandKeys), view)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: key is not valid for cmd '{}' (the cmd accepts: {})"),
                                         path_key(path, view), cmd, key_list(allowed)));
        } else {
            ctx.diag.warning(luisa::format(FMT_STRING("{}: unknown key"), path_key(path, view)));
        }
    }
}

// ---------------------------------------------------------------------------
// field helpers
// ---------------------------------------------------------------------------

void field_bool(Ctx &ctx, const yyjson_val *object, const char *key,
                luisa::string_view path, bool &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_bool(ctx, value, path_key(path, key))) { out = *parsed; }
    }
}

void field_bool_optional(Ctx &ctx, const yyjson_val *object, const char *key,
                         luisa::string_view path, bool &out, bool &has) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_bool(ctx, value, path_key(path, key))) {
            out = *parsed;
            has = true;
        }
    }
}

void field_u32(Ctx &ctx, const yyjson_val *object, const char *key,
               luisa::string_view path, uint32_t &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_u32(ctx, value, path_key(path, key))) { out = *parsed; }
    }
}

void field_u32_optional(Ctx &ctx, const yyjson_val *object, const char *key,
                        luisa::string_view path, uint32_t &out, bool &has) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_u32(ctx, value, path_key(path, key))) {
            out = *parsed;
            has = true;
        }
    }
}

void field_u64(Ctx &ctx, const yyjson_val *object, const char *key,
               luisa::string_view path, uint64_t &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_u64(ctx, value, path_key(path, key))) { out = *parsed; }
    }
}

void field_size(Ctx &ctx, const yyjson_val *object, const char *key,
                luisa::string_view path, size_t &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_size(ctx, value, path_key(path, key))) { out = *parsed; }
    }
}

void field_size_optional(Ctx &ctx, const yyjson_val *object, const char *key,
                         luisa::string_view path, size_t &out, bool &has) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_size(ctx, value, path_key(path, key))) {
            out = *parsed;
            has = true;
        }
    }
}

void field_f32(Ctx &ctx, const yyjson_val *object, const char *key,
               luisa::string_view path, float &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_f32(ctx, value, path_key(path, key))) { out = *parsed; }
    }
}

void field_string(Ctx &ctx, const yyjson_val *object, const char *key,
                  luisa::string_view path, luisa::string &out) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_string(ctx, value, path_key(path, key))) { out = std::move(*parsed); }
    }
}

void field_uint3(Ctx &ctx, const yyjson_val *object, const char *key,
                 luisa::string_view path, uint3 &out, bool allow_two_components = false) noexcept {
    if (auto *value = yyjson_obj_get(object, key)) {
        if (auto parsed = expect_uint3(ctx, value, path_key(path, key), allow_two_components)) {
            out = *parsed;
        }
    }
}

// `offset`/`size`/`src_offset`/`dst_offset` are dual-form: a 3-element array is
// the texture/volume region (uint3), a bare number the byte form (size_t).  See
// the corresponding field comment in native_shader_dispatch.h.
void field_region(Ctx &ctx, const yyjson_val *object, const char *key, luisa::string_view path,
                  uint3 &region, size_t &bytes) noexcept {
    auto *value = yyjson_obj_get(object, key);
    if (value == nullptr) { return; }
    auto field_path = path_key(path, key);
    if (yyjson_is_arr(value)) {
        if (auto parsed = expect_uint3(ctx, value, field_path, false)) { region = *parsed; }
    } else if (auto parsed = expect_size(ctx, value, field_path)) {
        bytes = *parsed;
    }
}

void field_string_array(Ctx &ctx, const yyjson_val *object, const char *key,
                        luisa::string_view path, luisa::vector<luisa::string> &out) noexcept {
    auto *value = yyjson_obj_get(object, key);
    if (value == nullptr) { return; }
    auto field_path = path_key(path, key);
    auto *array = expect_array(ctx, value, field_path);
    if (array == nullptr) { return; }
    out.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        if (auto parsed = expect_string(ctx, yyjson_arr_get(array, i), path_index(field_path, i))) {
            out.emplace_back(std::move(*parsed));
        }
    }
}

// An enumerating string field: the canonical (lowercase, '_') spelling of the
// value is stored, so the writer round-trips it unchanged.
template<size_t N>
void field_enum(Ctx &ctx, const yyjson_val *object, const char *key, luisa::string_view path,
                const Spelling (&table)[N], const char *what, luisa::string &out) noexcept {
    auto *value = yyjson_obj_get(object, key);
    if (value == nullptr) { return; }
    auto field_path = path_key(path, key);
    if (!yyjson_is_str(value)) {
        ctx.diag.error(expected_message(field_path, what, value));
        return;
    }
    auto text = luisa::string_view{yyjson_get_str(value), yyjson_get_len(value)};
    auto canonical = canonical_spelling(text);
    if (find_spelling(table, canonical) == nullptr) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: unknown {} '{}' (expected one of {})"),
                                     field_path, what, text, spelling_list(table)));
        return;
    }
    out = std::move(canonical);
}

// ---------------------------------------------------------------------------
// readers: mode / config
// ---------------------------------------------------------------------------

[[nodiscard]] InputJson parse_input(Ctx &ctx, const yyjson_val *value,
                                    luisa::string_view path) noexcept {
    auto result = InputJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kInputKeys), {});
    auto has_file = yyjson_obj_get(object, "file") != nullptr;
    auto has_inline = yyjson_obj_get(object, "inline") != nullptr;
    auto has_resource = yyjson_obj_get(object, "resource") != nullptr;
    auto source_count = (has_file ? 1 : 0) + (has_inline ? 1 : 0) + (has_resource ? 1 : 0);
    if (source_count > 1) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: an input names exactly one of 'file', 'inline' or 'resource'"),
                                     path));
    }
    // `decode_hex` reports its own message (length, digit and limit problems);
    // the caller only prefixes the path.
    auto decode_inline = [&ctx, &result](luisa::string_view text, luisa::string_view hex_path) noexcept {
        auto error = luisa::string{};
        if (!decode_hex(text, ctx.limits.max_inline_bytes, result.inline_bytes, error)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: {}"), hex_path, error));
        }
    };
    if (has_file) {
        result.kind = InputJson::Kind::File;
        field_string(ctx, object, "file", path, result.file);
    } else if (has_resource) {
        result.kind = InputJson::Kind::Resource;
        field_string(ctx, object, "resource", path, result.resource);
    } else if (has_inline) {
        result.kind = InputJson::Kind::Inline;
        auto *inline_value = yyjson_obj_get(object, "inline");
        auto inline_path = path_key(path, "inline");
        if (yyjson_is_str(inline_value)) {
            // The documented form is {"inline": {"hex": "..."}}; a bare hex
            // string is accepted as well, and canonicalised by the writer.
            if (auto text = expect_string(ctx, inline_value, inline_path)) {
                decode_inline(view_of(*text), inline_path);
            }
        } else if (auto *inline_object = expect_object(ctx, inline_value, inline_path)) {
            if (!check_depth(ctx, inline_path)) { return result; }
            auto inline_guard = DepthGuard{ctx};
            check_object_keys(ctx, inline_object, inline_path, keys_of(kInlineKeys), {});
            auto *hex_value = yyjson_obj_get(inline_object, "hex");
            if (hex_value == nullptr) {
                ctx.diag.error(luisa::format(FMT_STRING("{}.hex: inline input requires a hex payload"),
                                             inline_path));
            } else if (auto text = expect_string(ctx, hex_value, path_key(inline_path, "hex"))) {
                decode_inline(view_of(*text), path_key(inline_path, "hex"));
            }
        }
    }
    field_size(ctx, object, "offset", path, result.offset);
    field_size(ctx, object, "size", path, result.size);
    field_enum(ctx, object, "compression", path, kCompressionSpellings,
               "compression", result.compression);
    return result;
}

[[nodiscard]] OutputJson parse_output(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = OutputJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kOutputKeys), {});
    auto has_discard = yyjson_obj_get(object, "discard") != nullptr;
    field_bool(ctx, object, "discard", path, result.discard);
    field_string(ctx, object, "file", path, result.file);
    field_enum(ctx, object, "format", path, kOutputFormatSpellings, "output format", result.format);
    field_bool(ctx, object, "overwrite", path, result.overwrite);
    // Naming a file is what makes a sink write; `"discard": true` (or a sink
    // without a file) keeps the payload on the host, which is what the README
    // documents.
    if (!has_discard) { result.discard = result.file.empty(); }
    return result;
}

[[nodiscard]] VerifyJson parse_verify(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = VerifyJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kVerifyKeys), {});
    field_enum(ctx, object, "kind", path, kVerifyKindSpellings, "verify kind", result.kind);
    field_string(ctx, object, "source", path, result.source);
    field_f32(ctx, object, "k", path, result.k);
    field_f32(ctx, object, "c", path, result.c);
    field_f32(ctx, object, "tolerance", path, result.tolerance);
    return result;
}

[[nodiscard]] WindowJson parse_window(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = WindowJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kWindowKeys), {});
    field_string(ctx, object, "title", path, result.title);
    field_u32(ctx, object, "width", path, result.width);
    field_u32(ctx, object, "height", path, result.height);
    field_bool(ctx, object, "vsync", path, result.vsync);
    return result;
}

[[nodiscard]] SnapshotJson parse_snapshot(Ctx &ctx, const yyjson_val *value,
                                          luisa::string_view path) noexcept {
    auto result = SnapshotJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kSnapshotKeys), {});
    field_u32(ctx, object, "every", path, result.every);
    field_string(ctx, object, "path", path, result.path);
    return result;
}

[[nodiscard]] ModeJson parse_mode(Ctx &ctx, const yyjson_val *value,
                                  luisa::string_view path) noexcept {
    auto result = ModeJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kModeKeys), {});
    if (auto *type_value = yyjson_obj_get(object, "type")) {
        auto type_path = path_key(path, "type");
        if (auto text = expect_string(ctx, type_value, type_path)) {
            auto canonical = canonical_spelling(*text);
            auto *spelling = find_spelling(kModeTypeSpellings, canonical);
            if (spelling == nullptr) {
                ctx.diag.error(luisa::format(FMT_STRING("{}: unknown mode type '{}' (expected one of {})"),
                                             type_path, *text, spelling_list(kModeTypeSpellings)));
            } else {
                result.interactive = spelling->value != 0u;
            }
        }
    }
    field_u32(ctx, object, "frames", path, result.frames);
    field_bool(ctx, object, "gui", path, result.gui);
    if (auto *window = yyjson_obj_get(object, "window")) {
        result.window = parse_window(ctx, window, path_key(path, "window"));
    }
    field_string(ctx, object, "display_image", path, result.display_image);
    field_string(ctx, object, "display_destination", path, result.display_destination);
    field_f32(ctx, object, "display_scale", path, result.display_scale);
    field_string(ctx, object, "display_kernel", path, result.display_kernel);
    field_bool(ctx, object, "dispatch_per_frame", path, result.dispatch_per_frame);
    field_u32(ctx, object, "exit_after_frames", path, result.exit_after_frames);
    if (auto *snapshot = yyjson_obj_get(object, "snapshot")) {
        result.snapshot = parse_snapshot(ctx, snapshot, path_key(path, "snapshot"));
    }
    return result;
}

[[nodiscard]] DStorageConfigJson parse_dstorage(Ctx &ctx, const yyjson_val *value,
                                                luisa::string_view path) noexcept {
    auto result = DStorageConfigJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kDStorageKeys), {});
    field_bool(ctx, object, "enabled", path, result.enabled);
    field_size(ctx, object, "staging_buffer_size", path, result.staging_buffer_size);
    field_enum(ctx, object, "compression", path, kCompressionSpellings,
               "compression", result.compression);
    return result;
}

[[nodiscard]] JsonLimits parse_limits(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = JsonLimits{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kLimitKeys), {});
    field_size(ctx, object, "max_document_bytes", path, result.max_document_bytes);
    field_size(ctx, object, "max_resources", path, result.max_resources);
    field_size(ctx, object, "max_shaders", path, result.max_shaders);
    field_size(ctx, object, "max_commands", path, result.max_commands);
    field_size(ctx, object, "max_inline_bytes", path, result.max_inline_bytes);
    field_size(ctx, object, "max_uniform_bytes", path, result.max_uniform_bytes);
    field_size(ctx, object, "max_bindings_per_dispatch", path, result.max_bindings_per_dispatch);
    field_size(ctx, object, "max_resource_bytes", path, result.max_resource_bytes);
    field_size(ctx, object, "max_string_bytes", path, result.max_string_bytes);
    field_size(ctx, object, "max_errors", path, result.max_errors);
    field_u32(ctx, object, "max_depth", path, result.max_depth);
    return result;
}

[[nodiscard]] ConfigJson parse_config(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = ConfigJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kConfigKeys), {});
    field_string(ctx, object, "backend", path, result.backend);
    field_enum(ctx, object, "default_language", path, kLanguageSpellings,
               "language", result.default_language);
    field_u32(ctx, object, "shader_model", path, result.shader_model);
    field_bool(ctx, object, "optimize", path, result.optimize);
    field_bool(ctx, object, "fast_math", path, result.fast_math);
    field_bool(ctx, object, "debug_info", path, result.debug_info);
    field_uint3(ctx, object, "block_size", path, result.block_size);
    if (auto *push_constant_size = yyjson_obj_get(object, "push_constant_size")) {
        // `null` (the default) means "reflect per shader".
        if (yyjson_is_null(push_constant_size)) {
            result.has_push_constant_size = false;
        } else if (auto parsed = expect_u32(ctx, push_constant_size,
                                            path_key(path, "push_constant_size"))) {
            result.push_constant_size = *parsed;
            result.has_push_constant_size = true;
        }
    }
    field_string_array(ctx, object, "include_dirs", path, result.include_dirs);
    field_string(ctx, object, "output_dir", path, result.output_dir);
    if (auto *dstorage = yyjson_obj_get(object, "dstorage")) {
        result.dstorage = parse_dstorage(ctx, dstorage, path_key(path, "dstorage"));
    }
    field_bool(ctx, object, "strict", path, result.strict);
    field_enum(ctx, object, "log_level", path, kLogLevelSpellings, "log level", result.log_level);
    if (auto *limits = yyjson_obj_get(object, "limits")) {
        result.limits = parse_limits(ctx, limits, path_key(path, "limits"));
    }
    return result;
}

[[nodiscard]] ShaderJson parse_shader(Ctx &ctx, const yyjson_val *value,
                                      luisa::string_view path) noexcept {
    auto result = ShaderJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kShaderKeys), {});
    field_string(ctx, object, "name", path, result.name);
    if (auto *language = yyjson_obj_get(object, "language")) {
        auto language_path = path_key(path, "language");
        if (auto text = expect_string(ctx, language, language_path)) {
            auto canonical = canonical_spelling(*text);
            auto *spelling = find_spelling(kLanguageSpellings, canonical);
            if (spelling == nullptr) {
                ctx.diag.error(luisa::format(FMT_STRING("{}: unknown language '{}' (expected one of {})"),
                                             language_path, *text, spelling_list(kLanguageSpellings)));
            } else {
                result.language = static_cast<NativeShaderLanguage>(spelling->value);
                result.has_language = true;
            }
        }
    }
    auto has_path = yyjson_obj_get(object, "path") != nullptr;
    auto has_source = yyjson_obj_get(object, "source") != nullptr;
    if (has_path && has_source) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: a shader names either 'path' or 'source', not both"), path));
    }
    field_string(ctx, object, "path", path, result.path);
    field_string(ctx, object, "source", path, result.source);
    auto source_type_is_file = false;
    auto has_source_type = false;
    if (auto *source_type = yyjson_obj_get(object, "source_type")) {
        auto source_type_path = path_key(path, "source_type");
        if (auto text = expect_string(ctx, source_type, source_type_path)) {
            auto canonical = canonical_spelling(*text);
            auto *spelling = find_spelling(kSourceTypeSpellings, canonical);
            if (spelling == nullptr) {
                ctx.diag.error(luisa::format(FMT_STRING("{}: unknown source type '{}' (expected one of {})"),
                                             source_type_path, *text, spelling_list(kSourceTypeSpellings)));
            } else {
                source_type_is_file = spelling->value != 0u;
                has_source_type = true;
            }
        }
    }
    result.source_is_file = has_path || (!has_source && has_source_type && source_type_is_file);
    field_string(ctx, object, "entry_point", path, result.entry_point);
    field_uint3(ctx, object, "block_size", path, result.block_size);
    field_u32(ctx, object, "push_constant_size", path, result.push_constant_size);
    field_string_array(ctx, object, "include_dirs", path, result.include_dirs);
    field_bool(ctx, object, "optimize", path, result.optimize);
    field_bool(ctx, object, "fast_math", path, result.fast_math);
    field_bool(ctx, object, "debug_info", path, result.debug_info);
    return result;
}

// ---------------------------------------------------------------------------
// readers: storage, resources
// ---------------------------------------------------------------------------

// `storage` is shared by resources and by the texture copy commands; a
// block-compressed name gets its dedicated diagnostic instead of the generic
// "unknown pixel storage" one.
void field_storage(Ctx &ctx, const yyjson_val *object, const char *key,
                   luisa::string_view path, luisa::string &out) noexcept {
    auto *value = yyjson_obj_get(object, key);
    if (value == nullptr) { return; }
    auto field_path = path_key(path, key);
    if (auto text = expect_string(ctx, value, field_path)) {
        auto canonical = canonical_spelling(*text);
        if (canonical.starts_with("bc") || canonical.starts_with("astc")) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: block-compressed storage '{}' is not supported"),
                                         field_path, *text));
            return;
        }
        if (find_spelling(kStorageSpellings, canonical) == nullptr) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: unknown pixel storage '{}' (expected one of {})"),
                                         field_path, *text, spelling_list(kStorageSpellings)));
            return;
        }
        out = std::move(canonical);
    }
}

[[nodiscard]] KeySet resource_key_set(ResourceType type) noexcept {
    switch (type) {
        case ResourceType::Buffer: return keys_of(kBufferResourceKeys);
        case ResourceType::Texture: return keys_of(kTextureResourceKeys);
        case ResourceType::Volume: return keys_of(kVolumeResourceKeys);
        case ResourceType::BindlessArray: return keys_of(kBindlessResourceKeys);
        case ResourceType::Accel: return keys_of(kCommonResourceKeys);
        case ResourceType::Mesh: return keys_of(kMeshResourceKeys);
        case ResourceType::ProceduralPrimitive: return keys_of(kProceduralPrimitiveResourceKeys);
    }
    return KeySet{};
}

[[nodiscard]] ResourceJson parse_resource(Ctx &ctx, const yyjson_val *value,
                                          luisa::string_view path) noexcept {
    auto result = ResourceJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    // The type selects the allowed key set, so it is read first.
    auto *type_value = yyjson_obj_get(object, "type");
    if (type_value == nullptr) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: missing 'type'"), path));
    } else {
        auto type_path = path_key(path, "type");
        if (auto text = expect_string(ctx, type_value, type_path)) {
            auto canonical = canonical_spelling(*text);
            auto *spelling = find_spelling(kResourceTypeSpellings, canonical);
            if (spelling == nullptr) {
                // A retired name gets its own diagnostic: a resource type no
                // backend implements is a different thing from a typo.
                auto unsupported = false;
                for (auto name : unsupported_resource_types()) {
                    if (canonical != name) { continue; }
                    unsupported = true;
                    if (name == "indirect_dispatch_buffer") {
                        ctx.diag.error(luisa::format(FMT_STRING("{}: '{}' is not supported: indirect dispatch is unsupported by this example on every backend, so the resource has no use here"),
                                                     type_path, name));
                    } else {
                        ctx.diag.error(luisa::format(FMT_STRING("{}: '{}' is not supported: no backend implements curve or motion-blur acceleration structures, so this example rejects them everywhere"),
                                                     type_path, name));
                    }
                    break;
                }
                if (!unsupported) {
                    ctx.diag.error(luisa::format(FMT_STRING("{}: unknown resource type '{}' (expected one of {})"),
                                                 type_path, *text, spelling_list(kResourceTypeSpellings)));
                }
            } else {
                result.type = static_cast<ResourceType>(spelling->value);
            }
        }
    }
    check_object_keys(ctx, object, path, resource_key_set(result.type), {});
    field_string(ctx, object, "name", path, result.name);
    switch (result.type) {
        case ResourceType::Buffer: {
            field_enum(ctx, object, "element", path, kBufferElementSpellings,
                       "buffer element type", result.element);
            field_size(ctx, object, "count", path, result.count);
            field_size_optional(ctx, object, "byte_size", path, result.byte_size, result.has_byte_size);
            break;
        }
        case ResourceType::Texture:
        case ResourceType::Volume: {
            field_storage(ctx, object, "storage", path, result.storage);
            // A texture may give [w, h] (z is then the implied 1); a volume
            // always needs [w, h, d].
            field_uint3(ctx, object, "size", path, result.size,
                        result.type == ResourceType::Texture);
            field_u32(ctx, object, "levels", path, result.levels);
            field_enum(ctx, object, "element", path, kChannelElementSpellings,
                       "channel element type", result.element);
            if (result.element.empty() && !result.storage.empty()) {
                result.element = default_channel_element(result.storage);
            }
            break;
        }
        case ResourceType::BindlessArray: {
            field_size(ctx, object, "slot_count", path, result.slot_count);
            field_enum(ctx, object, "slot_type", path, kSlotTypeSpellings,
                       "bindless slot type", result.slot_type);
            break;
        }
        case ResourceType::Mesh: {
            field_string(ctx, object, "vertex_buffer", path, result.vertex_buffer);
            field_string(ctx, object, "triangle_buffer", path, result.triangle_buffer);
            break;
        }
        case ResourceType::ProceduralPrimitive: {
            field_string(ctx, object, "aabb_buffer", path, result.aabb_buffer);
            break;
        }
        case ResourceType::Accel: {
            break;
        }
    }
    if (auto *input = yyjson_obj_get(object, "input")) {
        result.input = parse_input(ctx, input, path_key(path, "input"));
    }
    return result;
}

[[nodiscard]] luisa::vector<ResourceJson> parse_resources(Ctx &ctx, const yyjson_val *value,
                                                          luisa::string_view path) noexcept {
    auto result = luisa::vector<ResourceJson>{};
    auto *array = expect_array(ctx, value, path);
    if (array == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    result.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        result.emplace_back(parse_resource(ctx, yyjson_arr_get(array, i), path_index(path, i)));
    }
    return result;
}

[[nodiscard]] luisa::vector<ShaderJson> parse_shaders(Ctx &ctx, const yyjson_val *value,
                                                      luisa::string_view path) noexcept {
    auto result = luisa::vector<ShaderJson>{};
    auto *array = expect_array(ctx, value, path);
    if (array == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    result.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        result.emplace_back(parse_shader(ctx, yyjson_arr_get(array, i), path_index(path, i)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// readers: workflow
// ---------------------------------------------------------------------------

[[nodiscard]] SamplerJson parse_sampler(Ctx &ctx, const yyjson_val *value,
                                        luisa::string_view path) noexcept {
    auto result = SamplerJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kSamplerKeys), {});
    field_enum(ctx, object, "filter", path, kFilterSpellings, "sampler filter", result.filter);
    field_enum(ctx, object, "address", path, kAddressSpellings, "sampler address", result.address);
    return result;
}

[[nodiscard]] BindingJson parse_binding(Ctx &ctx, const yyjson_val *value,
                                        luisa::string_view path) noexcept {
    auto result = BindingJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kBindingKeys), {});
    field_u32_optional(ctx, object, "index", path, result.index, result.has_index);
    field_u32_optional(ctx, object, "register", path, result.reg, result.has_register);
    field_u32(ctx, object, "space", path, result.space);
    field_string(ctx, object, "resource", path, result.resource);
    field_size(ctx, object, "offset", path, result.offset);
    field_size(ctx, object, "size", path, result.size);
    field_enum(ctx, object, "usage", path, kUsageSpellings, "usage", result.usage);
    return result;
}

[[nodiscard]] BindlessModJson parse_modification(Ctx &ctx, const yyjson_val *value,
                                                 luisa::string_view path) noexcept {
    auto result = BindlessModJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kModificationKeys), {});
    field_u32(ctx, object, "slot", path, result.slot);
    field_enum(ctx, object, "kind", path, kBindlessKindSpellings,
               "bindless modification kind", result.kind);
    field_enum(ctx, object, "op", path, kBindlessOpSpellings, "bindless operation", result.op);
    field_string(ctx, object, "resource", path, result.resource);
    field_size(ctx, object, "offset", path, result.offset);
    field_size(ctx, object, "size", path, result.size);
    if (auto *sampler = yyjson_obj_get(object, "sampler")) {
        result.sampler = parse_sampler(ctx, sampler, path_key(path, "sampler"));
    }
    return result;
}

[[nodiscard]] AccelModJson parse_accel_modification(Ctx &ctx, const yyjson_val *value,
                                                    luisa::string_view path) noexcept {
    auto result = AccelModJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kAccelModificationKeys), {});
    field_u32(ctx, object, "index", path, result.index);
    field_u32_optional(ctx, object, "user_id", path, result.user_id, result.has_user_id);
    field_u32_optional(ctx, object, "visibility", path, result.visibility, result.has_visibility);
    field_bool_optional(ctx, object, "opaque", path, result.opaque, result.has_opaque);
    if (auto *transform = yyjson_obj_get(object, "transform")) {
        if (expect_pod_array(ctx, transform, path_key(path, "transform"),
                             result.transform, "numbers")) {
            result.has_transform = true;
        }
    }
    if (yyjson_obj_get(object, "primitive") != nullptr) {
        field_string(ctx, object, "primitive", path, result.primitive);
        result.has_primitive = true;
    }
    return result;
}

// `type` (a uniform spelling) implies the payload layout; the value is either a
// single number, a 2..4 element array of numbers, or a hex string.  Every
// component is stored little-endian.
[[nodiscard]] bool uniform_type_layout(luisa::string_view canonical,
                                       uint32_t &components) noexcept {
    if (canonical == "hex") {
        components = 0u;
        return true;
    }
    auto prefix = size_t{0u};
    if (canonical.starts_with("float32")) {
        prefix = 7u;
    } else if (canonical.starts_with("uint32") || canonical.starts_with("int32")) {
        prefix = 6u;
    } else {
        return false;
    }
    auto rest = canonical.substr(prefix);
    if (rest.empty()) {
        components = 1u;
        return true;
    }
    if (rest.size() == 2u && rest[0] == 'x' && rest[1] >= '2' && rest[1] <= '4') {
        components = static_cast<uint32_t>(rest[1] - '0');
        return true;
    }
    return false;
}

void push_u32_le(luisa::vector<std::byte> &out, uint32_t bits) noexcept {
    for (auto shift : {0u, 8u, 16u, 24u}) {
        out.emplace_back(static_cast<std::byte>(
            static_cast<unsigned char>((bits >> shift) & 0xffu)));
    }
}

[[nodiscard]] bool encode_uniform_value(Ctx &ctx, const yyjson_val *value,
                                        luisa::string_view path,
                                        luisa::string_view canonical_type,
                                        uint32_t components,
                                        luisa::vector<std::byte> &out) noexcept {
    out.clear();
    auto is_float = canonical_type.starts_with("float");
    auto is_unsigned = canonical_type.starts_with("uint");
    auto push_element = [&](const yyjson_val *element, luisa::string_view element_path) noexcept {
        if (is_float) {
            auto parsed = expect_f32(ctx, element, element_path);
            if (!parsed) { return false; }
            auto bits = uint32_t{0u};
            std::memcpy(&bits, &*parsed, sizeof(bits));
            push_u32_le(out, bits);
            return true;
        }
        if (is_unsigned) {
            auto parsed = expect_u32(ctx, element, element_path);
            if (!parsed) { return false; }
            push_u32_le(out, *parsed);
            return true;
        }
        auto parsed = expect_i64(ctx, element, element_path);
        if (!parsed) { return false; }
        if (*parsed < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
            *parsed > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: value {} is out of range for int32"),
                                         element_path, *parsed));
            return false;
        }
        push_u32_le(out, static_cast<uint32_t>(static_cast<int32_t>(*parsed)));
        return true;
    };
    if (components == 1u) {
        auto *element = value;
        if (yyjson_is_arr(value)) {
            if (yyjson_arr_size(value) != 1u) {
                ctx.diag.error(luisa::format(FMT_STRING("{}: expected a single number, got {} elements"),
                                             path, yyjson_arr_size(value)));
                return false;
            }
            element = yyjson_arr_get(value, 0u);
        }
        return push_element(element, path);
    }
    if (!yyjson_is_arr(value)) {
        ctx.diag.error(expected_message(path, "array of numbers", value));
        return false;
    }
    if (yyjson_arr_size(value) != components) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: expected a {}-element array of numbers, got {}"),
                                     path, components, yyjson_arr_size(value)));
        return false;
    }
    for (auto i = uint32_t{0u}; i < components; i++) {
        if (!push_element(yyjson_arr_get(value, i), path_index(path, i))) { return false; }
    }
    return true;
}

[[nodiscard]] bool parse_uniform_payload(Ctx &ctx, const yyjson_val *object,
                                         luisa::string_view path, luisa::string &type,
                                         luisa::vector<std::byte> &bytes,
                                         size_t &alignment) noexcept {
    auto has_type = false;
    if (auto *type_value = yyjson_obj_get(object, "type")) {
        auto type_path = path_key(path, "type");
        if (auto text = expect_string(ctx, type_value, type_path)) {
            auto canonical = canonical_spelling(*text);
            auto components = uint32_t{0u};
            if (!uniform_type_layout(canonical, components)) {
                ctx.diag.error(luisa::format(FMT_STRING("{}: unknown uniform type '{}' (expected one of {})"),
                                             type_path, *text, spelling_list(kUniformTypeSpellings)));
            } else {
                type = std::move(canonical);
                alignment = components == 0u ? size_t{4u} : static_cast<size_t>(components) * 4u;
                has_type = true;
            }
        }
    }
    if (!has_type) {
        if (yyjson_obj_get(object, "hex") == nullptr) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: a uniform payload needs 'type' with 'value', or 'hex'"),
                                         path));
            return false;
        }
        type = "hex";
        alignment = 4u;
    }
    if (type == "hex") {
        auto *hex_value = yyjson_obj_get(object, "hex");
        if (hex_value == nullptr) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: a hex uniform payload requires 'hex'"), path));
            return false;
        }
        auto hex_path = path_key(path, "hex");
        auto text = expect_string(ctx, hex_value, hex_path);
        if (!text) { return false; }
        auto error = luisa::string{};
        if (!decode_hex(view_of(*text), ctx.limits.max_inline_bytes, bytes, error)) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: {}"), hex_path, error));
            return false;
        }
        return true;
    }
    auto *value = yyjson_obj_get(object, "value");
    if (value == nullptr) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: a uniform of type '{}' requires a 'value'"),
                                     path, type));
        return false;
    }
    auto components = uint32_t{0u};
    (void)uniform_type_layout(type, components);
    return encode_uniform_value(ctx, value, path_key(path, "value"), type, components, bytes);
}

[[nodiscard]] UniformJson parse_uniform(Ctx &ctx, const yyjson_val *value,
                                        luisa::string_view path) noexcept {
    auto result = UniformJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kUniformKeys), {});
    (void)parse_uniform_payload(ctx, object, path, result.type, result.bytes, result.alignment);
    return result;
}

[[nodiscard]] ArgumentJson parse_argument(Ctx &ctx, const yyjson_val *value,
                                          luisa::string_view path) noexcept {
    auto result = ArgumentJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, object, path, keys_of(kArgumentKeys), {});
    auto *kind_value = yyjson_obj_get(object, "kind");
    if (kind_value == nullptr) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: missing 'kind'"), path));
        return result;
    }
    auto kind_path = path_key(path, "kind");
    if (auto text = expect_string(ctx, kind_value, kind_path)) {
        auto canonical = canonical_spelling(*text);
        if (find_spelling(kArgumentKindSpellings, canonical) == nullptr) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: unknown argument kind '{}' (expected one of {})"),
                                         kind_path, *text, spelling_list(kArgumentKindSpellings)));
        } else {
            result.kind = std::move(canonical);
        }
    }
    field_string(ctx, object, "resource", path, result.resource);
    field_size(ctx, object, "offset", path, result.offset);
    field_u32(ctx, object, "level", path, result.level);
    auto has_payload = yyjson_obj_get(object, "type") != nullptr ||
                       yyjson_obj_get(object, "value") != nullptr ||
                       yyjson_obj_get(object, "hex") != nullptr;
    if (result.kind == "uniform") {
        auto type = luisa::string{};
        (void)parse_uniform_payload(ctx, object, path, type, result.bytes, result.alignment);
    } else if (has_payload) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: 'type'/'value'/'hex' are only meaningful for a 'uniform' argument"),
                                     path));
    }
    return result;
}

template<typename T, typename Reader>
[[nodiscard]] luisa::vector<T> parse_array(Ctx &ctx, const yyjson_val *value, luisa::string_view path,
                                           Reader reader) noexcept {
    auto result = luisa::vector<T>{};
    auto *array = expect_array(ctx, value, path);
    if (array == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    result.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        result.emplace_back(reader(ctx, yyjson_arr_get(array, i), path_index(path, i)));
    }
    return result;
}

[[nodiscard]] luisa::vector<uint3> parse_batched(Ctx &ctx, const yyjson_val *value,
                                                 luisa::string_view path) noexcept {
    auto result = luisa::vector<uint3>{};
    auto *array = expect_array(ctx, value, path);
    if (array == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    result.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        if (auto parsed = expect_uint3(ctx, yyjson_arr_get(array, i), path_index(path, i), false)) {
            result.emplace_back(*parsed);
        }
    }
    return result;
}

// The field set shared by `native_dispatch` and the `native_shader_dispatch`
// alias of `custom_command`.
void parse_native_dispatch_fields(Ctx &ctx, const yyjson_val *object,
                                  luisa::string_view path,
                                  CommandJson &result) noexcept {
    field_string(ctx, object, "shader", path, result.shader);
    field_uint3(ctx, object, "dispatch", path, result.dispatch);
    field_uint3(ctx, object, "grid", path, result.grid);
    if (auto *bindings = yyjson_obj_get(object, "bindings")) {
        result.bindings = parse_array<BindingJson>(ctx, bindings, path_key(path, "bindings"),
                                                   parse_binding);
    }
    if (auto *uniforms = yyjson_obj_get(object, "uniforms")) {
        result.uniforms = parse_array<UniformJson>(ctx, uniforms, path_key(path, "uniforms"),
                                                   parse_uniform);
    }
    field_bool(ctx, object, "allow_usage_override", path, result.allow_usage_override);
}

[[nodiscard]] CommandJson parse_command(Ctx &ctx, const yyjson_val *value,
                                        luisa::string_view path) noexcept {
    auto result = CommandJson{};
    auto *object = expect_object(ctx, value, path);
    if (object == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    auto *cmd_value = yyjson_obj_get(object, "cmd");
    if (cmd_value == nullptr) {
        ctx.diag.error(luisa::format(FMT_STRING("{}: missing 'cmd'"), path));
        check_object_keys(ctx, object, path, keys_of(kAllCommandKeys), {});
        return result;
    }
    auto cmd_path = path_key(path, "cmd");
    if (!yyjson_is_str(cmd_value)) {
        ctx.diag.error(expected_message(cmd_path, "string", cmd_value));
        return result;
    }
    auto cmd_text = luisa::string_view{yyjson_get_str(cmd_value), yyjson_get_len(cmd_value)};
    auto cmd_canonical = canonical_spelling(cmd_text);
    auto *spelling = find_spelling(kCommandKindSpellings, cmd_canonical);
    if (spelling == nullptr) {
        // A retired name gets its own diagnostic: a command kind no backend
        // implements is a different thing from a typo.
        auto unsupported = false;
        for (auto name : unsupported_command_kinds()) {
            if (cmd_canonical != name) { continue; }
            unsupported = true;
            ctx.diag.error(luisa::format(FMT_STRING("{}: '{}' is not supported: no backend implements curve or motion-blur acceleration structures, so this example rejects them everywhere"),
                                         cmd_path, name));
            break;
        }
        if (!unsupported) {
            ctx.diag.error(luisa::format(FMT_STRING("{}: unknown cmd '{}' (expected one of {})"),
                                         cmd_path, cmd_text, spelling_list(kCommandKindSpellings)));
        }
        return result;
    }
    result.kind = static_cast<CommandKind>(spelling->value);
    check_object_keys(ctx, object, path, command_key_set(result.kind),
                      command_kind_name(result.kind));
    switch (result.kind) {
        case CommandKind::BufferUpload: {
            field_string(ctx, object, "resource", path, result.resource);
            field_size(ctx, object, "offset", path, result.offset);
            field_size(ctx, object, "size", path, result.size);
            if (auto *input = yyjson_obj_get(object, "input")) {
                result.input = parse_input(ctx, input, path_key(path, "input"));
            }
            break;
        }
        case CommandKind::BufferDownload: {
            field_string(ctx, object, "resource", path, result.resource);
            field_size(ctx, object, "offset", path, result.offset);
            field_size(ctx, object, "size", path, result.size);
            if (auto *output = yyjson_obj_get(object, "output")) {
                result.output = parse_output(ctx, output, path_key(path, "output"));
            }
            if (auto *verify = yyjson_obj_get(object, "verify")) {
                result.verify = parse_verify(ctx, verify, path_key(path, "verify"));
            }
            break;
        }
        case CommandKind::BufferCopy: {
            field_string(ctx, object, "src", path, result.src);
            field_size(ctx, object, "src_offset", path, result.src_offset);
            field_string(ctx, object, "dst", path, result.dst);
            field_size(ctx, object, "dst_offset", path, result.dst_offset);
            field_size(ctx, object, "size", path, result.size);
            break;
        }
        case CommandKind::TextureUpload: {
            field_string(ctx, object, "resource", path, result.resource);
            field_u32(ctx, object, "level", path, result.level);
            field_region(ctx, object, "offset", path, result.offset3, result.offset);
            // `texture_offset` is the spelling the plan's normative table uses; it is
            // accepted as an alias and normalised to `offset`.
            if (yyjson_obj_get(object, "offset") == nullptr) {
                field_region(ctx, object, "texture_offset", path, result.offset3, result.offset);
            }
            field_region(ctx, object, "size", path, result.size3, result.size);
            field_storage(ctx, object, "storage", path, result.storage);
            if (auto *input = yyjson_obj_get(object, "input")) {
                result.input = parse_input(ctx, input, path_key(path, "input"));
            }
            break;
        }
        case CommandKind::TextureDownload: {
            field_string(ctx, object, "resource", path, result.resource);
            field_u32(ctx, object, "level", path, result.level);
            field_region(ctx, object, "offset", path, result.offset3, result.offset);
            field_region(ctx, object, "size", path, result.size3, result.size);
            field_storage(ctx, object, "storage", path, result.storage);
            if (auto *output = yyjson_obj_get(object, "output")) {
                result.output = parse_output(ctx, output, path_key(path, "output"));
            }
            break;
        }
        case CommandKind::TextureCopy: {
            field_storage(ctx, object, "storage", path, result.storage);
            field_string(ctx, object, "src", path, result.src);
            field_string(ctx, object, "dst", path, result.dst);
            field_u32(ctx, object, "src_level", path, result.src_level);
            field_u32(ctx, object, "dst_level", path, result.dst_level);
            field_region(ctx, object, "size", path, result.size3, result.size);
            field_region(ctx, object, "src_offset", path, result.src_offset3, result.src_offset);
            field_region(ctx, object, "dst_offset", path, result.dst_offset3, result.dst_offset);
            break;
        }
        case CommandKind::BufferToTextureCopy:
        case CommandKind::TextureToBufferCopy: {
            field_string(ctx, object, "buffer", path, result.buffer);
            field_size(ctx, object, "buffer_offset", path, result.buffer_offset);
            field_string(ctx, object, "texture", path, result.texture);
            field_storage(ctx, object, "storage", path, result.storage);
            field_u32(ctx, object, "level", path, result.level);
            field_region(ctx, object, "size", path, result.size3, result.size);
            field_region(ctx, object, "offset", path, result.offset3, result.offset);
            break;
        }
        case CommandKind::NativeDispatch: {
            parse_native_dispatch_fields(ctx, object, path, result);
            break;
        }
        case CommandKind::ShaderDispatch: {
            field_string(ctx, object, "shader", path, result.shader);
            if (auto *arguments = yyjson_obj_get(object, "arguments")) {
                result.arguments = parse_array<ArgumentJson>(ctx, arguments, path_key(path, "arguments"),
                                                             parse_argument);
            }
            field_uint3(ctx, object, "dispatch", path, result.dispatch);
            // `indirect` is recognised - so that a document carrying one is told
            // what is wrong with it - but no backend implements it, so it is
            // rejected outright instead of being parsed into a model type.
            if (yyjson_obj_get(object, "indirect") != nullptr) {
                ctx.diag.error(luisa::format(FMT_STRING("{}.indirect: indirect dispatch is not supported by any backend; use 'dispatch' or 'batched'"),
                                             path));
            }
            if (auto *batched = yyjson_obj_get(object, "batched")) {
                result.batched = parse_batched(ctx, batched, path_key(path, "batched"));
            }
            break;
        }
        case CommandKind::BindlessArrayUpdate: {
            field_string(ctx, object, "resource", path, result.resource);
            field_enum(ctx, object, "mode", path, kSlotTypeSpellings, "bindless mode", result.mode);
            if (auto *modifications = yyjson_obj_get(object, "modifications")) {
                result.modifications = parse_array<BindlessModJson>(
                    ctx, modifications, path_key(path, "modifications"), parse_modification);
            }
            break;
        }
        case CommandKind::MeshBuild: {
            field_string(ctx, object, "resource", path, result.resource);
            field_enum(ctx, object, "request", path, kRequestSpellings, "build request", result.request);
            field_string(ctx, object, "vertex_buffer", path, result.vertex_buffer);
            field_size(ctx, object, "vertex_buffer_offset", path, result.vertex_buffer_offset);
            field_size(ctx, object, "vertex_buffer_size", path, result.vertex_buffer_size);
            field_u32(ctx, object, "vertex_stride", path, result.vertex_stride);
            field_string(ctx, object, "triangle_buffer", path, result.triangle_buffer);
            field_size(ctx, object, "triangle_buffer_offset", path, result.triangle_buffer_offset);
            field_size(ctx, object, "triangle_buffer_size", path, result.triangle_buffer_size);
            break;
        }
        case CommandKind::ProceduralPrimitiveBuild: {
            field_string(ctx, object, "resource", path, result.resource);
            field_enum(ctx, object, "request", path, kRequestSpellings, "build request", result.request);
            field_string(ctx, object, "aabb_buffer", path, result.aabb_buffer);
            field_size(ctx, object, "aabb_buffer_offset", path, result.aabb_buffer_offset);
            field_size(ctx, object, "aabb_buffer_size", path, result.aabb_buffer_size);
            break;
        }
        case CommandKind::AccelBuild: {
            field_string(ctx, object, "resource", path, result.resource);
            field_u32(ctx, object, "instance_count", path, result.instance_count);
            field_enum(ctx, object, "request", path, kRequestSpellings, "build request", result.request);
            field_bool(ctx, object, "update_instance_buffer_only", path, result.update_instance_buffer_only);
            if (auto *modifications = yyjson_obj_get(object, "modifications")) {
                result.accel_modifications = parse_array<AccelModJson>(
                    ctx, modifications, path_key(path, "modifications"), parse_accel_modification);
            }
            break;
        }
        case CommandKind::CustomCommand: {
            if (auto *uuid = yyjson_obj_get(object, "uuid")) {
                auto uuid_path = path_key(path, "uuid");
                if (yyjson_is_str(uuid)) {
                    if (auto text = expect_string(ctx, uuid, uuid_path)) {
                        auto canonical = canonical_spelling(*text);
                        if (auto *uuid_spelling = find_spelling(kUuidSpellings, canonical)) {
                            result.uuid = uuid_spelling->value;
                        } else {
                            ctx.diag.error(luisa::format(FMT_STRING("{}: unknown custom command '{}' (expected one of {})"),
                                                         uuid_path, *text, spelling_list(kUuidSpellings)));
                        }
                    }
                } else {
                    field_u64(ctx, object, "uuid", path, result.uuid);
                }
            } else {
                ctx.diag.error(luisa::format(FMT_STRING("{}: custom_command requires 'uuid'"), path));
            }
            field_string(ctx, object, "resource", path, result.resource);
            field_string(ctx, object, "buffer", path, result.buffer);
            field_size(ctx, object, "buffer_offset", path, result.buffer_offset);
            field_string(ctx, object, "texture", path, result.texture);
            field_storage(ctx, object, "storage", path, result.storage);
            field_u32(ctx, object, "level", path, result.level);
            field_size(ctx, object, "offset", path, result.offset);
            field_size(ctx, object, "size", path, result.size);
            if (auto *input = yyjson_obj_get(object, "input")) {
                result.input = parse_input(ctx, input, path_key(path, "input"));
            }
            if (auto *output = yyjson_obj_get(object, "output")) {
                result.output = parse_output(ctx, output, path_key(path, "output"));
            }
            field_string(ctx, object, "label", path, result.label);
            if (result.uuid ==
                luisa::to_underlying(compute::CustomCommandUUID::NATIVE_SHADER_DISPATCH)) {
                parse_native_dispatch_fields(ctx, object, path, result);
            }
            break;
        }
        case CommandKind::Log: {
            field_string(ctx, object, "message", path, result.message);
            break;
        }
        case CommandKind::Synchronize: {
            field_string(ctx, object, "label", path, result.label);
            break;
        }
    }
    return result;
}

[[nodiscard]] luisa::vector<CommandJson> parse_workflow(Ctx &ctx, const yyjson_val *value,
                                                        luisa::string_view path) noexcept {
    auto result = luisa::vector<CommandJson>{};
    auto *array = expect_array(ctx, value, path);
    if (array == nullptr || !check_depth(ctx, path)) { return result; }
    auto guard = DepthGuard{ctx};
    result.reserve(yyjson_arr_size(array));
    for (auto i = size_t{0u}; i < yyjson_arr_size(array); i++) {
        result.emplace_back(parse_command(ctx, yyjson_arr_get(array, i), path_index(path, i)));
    }
    return result;
}

// ---------------------------------------------------------------------------
// readers: root
// ---------------------------------------------------------------------------

[[nodiscard]] DispatchJson parse_root(Ctx &ctx, const yyjson_val *root) noexcept {
    auto result = DispatchJson{};
    auto guard = DepthGuard{ctx};
    check_object_keys(ctx, root, {}, keys_of(kRootKeys), {});
    if (auto *version = yyjson_obj_get(root, "version")) {
        if (auto parsed = expect_u32(ctx, version, "version")) {
            result.version = *parsed;
            // `version` is optional and means 1; anything else is rejected so
            // that a newer document is never read with older semantics.
            if (*parsed < 1u) {
                ctx.diag.error(luisa::format(FMT_STRING("version: expected a version of at least 1, got {}"),
                                             *parsed));
            } else if (*parsed > 1u) {
                ctx.diag.error(luisa::format(FMT_STRING("version: unsupported document version {} (this build reads version 1)"),
                                             *parsed));
            }
        }
    }
    if (auto *mode = yyjson_obj_get(root, "mode")) {
        result.mode = parse_mode(ctx, mode, "mode");
    }
    if (auto *config = yyjson_obj_get(root, "config")) {
        result.config = parse_config(ctx, config, "config");
    }
    if (auto *shaders = yyjson_obj_get(root, "shaders")) {
        result.shaders = parse_shaders(ctx, shaders, "shaders");
    }
    if (auto *resources = yyjson_obj_get(root, "resources")) {
        result.resources = parse_resources(ctx, resources, "resources");
    }
    if (auto *workflow = yyjson_obj_get(root, "workflow")) {
        result.workflow = parse_workflow(ctx, workflow, "workflow");
    }
    return result;
}

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------
// Keys are emitted in the order of the README tables (and of the header's field
// order), every resolved value - defaults included - is written, and only the
// keys a command's kind owns are emitted, so that
// `write -> parse -> write` reproduces the very same text.

struct WriteCtx {
    yyjson_mut_doc *doc{nullptr};
    luisa::string error;

    void fail(luisa::string_view what) noexcept {
        if (error.empty()) {
            error = luisa::format(FMT_STRING("failed to write '{}' (out of memory?)"), what);
        }
    }
};

[[nodiscard]] yyjson_mut_val *make_object(WriteCtx &w, luisa::string_view what) noexcept {
    auto *object = yyjson_mut_obj(w.doc);
    if (object == nullptr) { w.fail(what); }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_array(WriteCtx &w, luisa::string_view what) noexcept {
    auto *array = yyjson_mut_arr(w.doc);
    if (array == nullptr) { w.fail(what); }
    return array;
}

void add_uint(WriteCtx &w, yyjson_mut_val *object, const char *key, uint64_t value) noexcept {
    if (!yyjson_mut_obj_add_uint(w.doc, object, key, value)) { w.fail(key); }
}

void add_bool(WriteCtx &w, yyjson_mut_val *object, const char *key, bool value) noexcept {
    if (!yyjson_mut_obj_add_bool(w.doc, object, key, value)) { w.fail(key); }
}

void add_real(WriteCtx &w, yyjson_mut_val *object, const char *key, double value) noexcept {
    if (!yyjson_mut_obj_add_real(w.doc, object, key, value)) { w.fail(key); }
}

void add_string(WriteCtx &w, yyjson_mut_val *object, const char *key,
                luisa::string_view value) noexcept {
    if (!yyjson_mut_obj_add_strncpy(w.doc, object, key, value.data(), value.size())) {
        w.fail(key);
    }
}

void add_null(WriteCtx &w, yyjson_mut_val *object, const char *key) noexcept {
    if (!yyjson_mut_obj_add_null(w.doc, object, key)) { w.fail(key); }
}

// An empty string is never a valid spelling of an enumerating field, so such a
// key is omitted (the parser's default then applies again on the next read).
void add_enum(WriteCtx &w, yyjson_mut_val *object, const char *key,
              luisa::string_view value) noexcept {
    if (value.empty()) { return; }
    add_string(w, object, key, value);
}

void add_value(WriteCtx &w, yyjson_mut_val *object, const char *key,
               yyjson_mut_val *value) noexcept {
    if (value == nullptr) { return; }// the failure is already recorded
    if (!yyjson_mut_obj_add_val(w.doc, object, key, value)) { w.fail(key); }
}

[[nodiscard]] yyjson_mut_val *make_uint2(WriteCtx &w, uint2 value) noexcept {
    auto *array = make_array(w, "uint2");
    if (array == nullptr) { return nullptr; }
    for (auto component : {value.x, value.y}) {
        if (!yyjson_mut_arr_add_uint(w.doc, array, component)) { w.fail("uint2"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_uint3(WriteCtx &w, uint3 value) noexcept {
    auto *array = make_array(w, "uint3");
    if (array == nullptr) { return nullptr; }
    for (auto component : {value.x, value.y, value.z}) {
        if (!yyjson_mut_arr_add_uint(w.doc, array, component)) { w.fail("uint3"); }
    }
    return array;
}

template<size_t N>
[[nodiscard]] yyjson_mut_val *make_real_array(WriteCtx &w, const float (&values)[N]) noexcept {
    auto *array = make_array(w, "numbers");
    if (array == nullptr) { return nullptr; }
    for (auto i = size_t{0u}; i < N; i++) {
        if (!yyjson_mut_arr_add_real(w.doc, array, static_cast<double>(values[i]))) {
            w.fail("numbers");
        }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_string_array(WriteCtx &w, const luisa::vector<luisa::string> &values) noexcept {
    auto *array = make_array(w, "strings");
    if (array == nullptr) { return nullptr; }
    for (auto &value : values) {
        if (!yyjson_mut_arr_add_strncpy(w.doc, array, value.data(), value.size())) {
            w.fail("strings");
        }
    }
    return array;
}

[[nodiscard]] luisa::string_view language_name(NativeShaderLanguage language) noexcept {
    for (auto spelling : kLanguageSpellings) {
        if (spelling.value == luisa::to_underlying(language)) { return spelling.name; }
    }
    return {};
}

[[nodiscard]] yyjson_mut_val *make_window(WriteCtx &w, const WindowJson &window) noexcept {
    auto *object = make_object(w, "window");
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "title", view_of(window.title));
    add_uint(w, object, "width", window.width);
    add_uint(w, object, "height", window.height);
    add_bool(w, object, "vsync", window.vsync);
    return object;
}

[[nodiscard]] yyjson_mut_val *make_snapshot(WriteCtx &w, const SnapshotJson &snapshot) noexcept {
    auto *object = make_object(w, "snapshot");
    if (object == nullptr) { return nullptr; }
    add_uint(w, object, "every", snapshot.every);
    add_string(w, object, "path", view_of(snapshot.path));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_mode(WriteCtx &w, const ModeJson &mode) noexcept {
    auto *object = make_object(w, "mode");
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "type", mode.interactive ? "interactive" : "offline");
    add_uint(w, object, "frames", mode.frames);
    add_bool(w, object, "gui", mode.gui);
    add_value(w, object, "window", make_window(w, mode.window));
    add_string(w, object, "display_image", view_of(mode.display_image));
    add_string(w, object, "display_destination", view_of(mode.display_destination));
    add_real(w, object, "display_scale", static_cast<double>(mode.display_scale));
    add_string(w, object, "display_kernel", view_of(mode.display_kernel));
    add_bool(w, object, "dispatch_per_frame", mode.dispatch_per_frame);
    add_uint(w, object, "exit_after_frames", mode.exit_after_frames);
    add_value(w, object, "snapshot", make_snapshot(w, mode.snapshot));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_dstorage(WriteCtx &w, const DStorageConfigJson &dstorage) noexcept {
    auto *object = make_object(w, "dstorage");
    if (object == nullptr) { return nullptr; }
    add_bool(w, object, "enabled", dstorage.enabled);
    add_uint(w, object, "staging_buffer_size", dstorage.staging_buffer_size);
    add_enum(w, object, "compression", view_of(dstorage.compression));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_limits(WriteCtx &w, const JsonLimits &limits) noexcept {
    auto *object = make_object(w, "limits");
    if (object == nullptr) { return nullptr; }
    add_uint(w, object, "max_document_bytes", limits.max_document_bytes);
    add_uint(w, object, "max_resources", limits.max_resources);
    add_uint(w, object, "max_shaders", limits.max_shaders);
    add_uint(w, object, "max_commands", limits.max_commands);
    add_uint(w, object, "max_inline_bytes", limits.max_inline_bytes);
    add_uint(w, object, "max_uniform_bytes", limits.max_uniform_bytes);
    add_uint(w, object, "max_bindings_per_dispatch", limits.max_bindings_per_dispatch);
    add_uint(w, object, "max_resource_bytes", limits.max_resource_bytes);
    add_uint(w, object, "max_string_bytes", limits.max_string_bytes);
    add_uint(w, object, "max_errors", limits.max_errors);
    add_uint(w, object, "max_depth", limits.max_depth);
    return object;
}

[[nodiscard]] yyjson_mut_val *make_config(WriteCtx &w, const ConfigJson &config) noexcept {
    auto *object = make_object(w, "config");
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "backend", view_of(config.backend));
    add_enum(w, object, "default_language", view_of(config.default_language));
    add_uint(w, object, "shader_model", config.shader_model);
    add_bool(w, object, "optimize", config.optimize);
    add_bool(w, object, "fast_math", config.fast_math);
    add_bool(w, object, "debug_info", config.debug_info);
    add_value(w, object, "block_size", make_uint3(w, config.block_size));
    if (config.has_push_constant_size) {
        add_uint(w, object, "push_constant_size", config.push_constant_size);
    } else {
        // `null` means "reflect per shader".
        add_null(w, object, "push_constant_size");
    }
    add_value(w, object, "include_dirs", make_string_array(w, config.include_dirs));
    add_string(w, object, "output_dir", view_of(config.output_dir));
    add_value(w, object, "dstorage", make_dstorage(w, config.dstorage));
    add_bool(w, object, "strict", config.strict);
    add_enum(w, object, "log_level", view_of(config.log_level));
    add_value(w, object, "limits", make_limits(w, config.limits));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_shader(WriteCtx &w, const ShaderJson &shader) noexcept {
    auto *object = make_object(w, "shader");
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "name", view_of(shader.name));
    auto language = language_name(shader.language);
    if (language.empty()) {
        w.fail("shader language");
    } else {
        add_string(w, object, "language", language);
    }
    if (shader.source_is_file) {
        add_string(w, object, "path", view_of(shader.path));
    } else {
        add_string(w, object, "source", view_of(shader.source));
    }
    add_string(w, object, "entry_point", view_of(shader.entry_point));
    add_value(w, object, "block_size", make_uint3(w, shader.block_size));
    add_uint(w, object, "push_constant_size", shader.push_constant_size);
    add_value(w, object, "include_dirs", make_string_array(w, shader.include_dirs));
    add_bool(w, object, "optimize", shader.optimize);
    add_bool(w, object, "fast_math", shader.fast_math);
    add_bool(w, object, "debug_info", shader.debug_info);
    return object;
}

[[nodiscard]] yyjson_mut_val *make_input(WriteCtx &w, const InputJson &input) noexcept {
    auto *object = make_object(w, "input");
    if (object == nullptr) { return nullptr; }
    switch (input.kind) {
        case InputJson::Kind::None: {
            // Nothing named: an empty object keeps the key round-trippable.
            break;
        }
        case InputJson::Kind::File: {
            add_string(w, object, "file", view_of(input.file));
            add_uint(w, object, "offset", input.offset);
            add_uint(w, object, "size", input.size);
            add_enum(w, object, "compression", view_of(input.compression));
            break;
        }
        case InputJson::Kind::Inline: {
            auto *inline_object = make_object(w, "inline");
            if (inline_object == nullptr) { break; }
            auto hex = encode_hex(luisa::span<const std::byte>{input.inline_bytes});
            add_string(w, inline_object, "hex", view_of(hex));
            add_value(w, object, "inline", inline_object);
            break;
        }
        case InputJson::Kind::Resource: {
            add_string(w, object, "resource", view_of(input.resource));
            add_uint(w, object, "offset", input.offset);
            add_uint(w, object, "size", input.size);
            break;
        }
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_output(WriteCtx &w, const OutputJson &output) noexcept {
    auto *object = make_object(w, "output");
    if (object == nullptr) { return nullptr; }
    add_bool(w, object, "discard", output.discard);
    add_string(w, object, "file", view_of(output.file));
    add_enum(w, object, "format", view_of(output.format));
    add_bool(w, object, "overwrite", output.overwrite);
    return object;
}

[[nodiscard]] yyjson_mut_val *make_verify(WriteCtx &w, const VerifyJson &verify) noexcept {
    auto *object = make_object(w, "verify");
    if (object == nullptr) { return nullptr; }
    add_enum(w, object, "kind", view_of(verify.kind));
    add_string(w, object, "source", view_of(verify.source));
    add_real(w, object, "k", static_cast<double>(verify.k));
    add_real(w, object, "c", static_cast<double>(verify.c));
    add_real(w, object, "tolerance", static_cast<double>(verify.tolerance));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_resource(WriteCtx &w, const ResourceJson &resource) noexcept {
    auto *object = make_object(w, "resource");
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "name", view_of(resource.name));
    auto type_name = resource_type_name(resource.type);
    if (luisa::string_view{type_name} == "unknown") {
        w.fail("the type of a resource");
    } else {
        add_string(w, object, "type", type_name);
    }
    switch (resource.type) {
        case ResourceType::Buffer: {
            if (!resource.element.empty()) {
                add_enum(w, object, "element", view_of(resource.element));
                add_uint(w, object, "count", resource.count);
            } else {
                // `element` is empty: the buffer is sized in bytes.
                add_uint(w, object, "byte_size", resource.byte_size);
            }
            break;
        }
        case ResourceType::Texture:
        case ResourceType::Volume: {
            add_enum(w, object, "storage", view_of(resource.storage));
            // A texture is written as [w, h] (the README form) when its z is the
            // implied 1, and as [w, h, d] otherwise; a volume is always 3-D.
            if (resource.type == ResourceType::Texture && resource.size.z == 1u) {
                add_value(w, object, "size", make_uint2(w, uint2{resource.size.x, resource.size.y}));
            } else {
                add_value(w, object, "size", make_uint3(w, resource.size));
            }
            add_uint(w, object, "levels", resource.levels);
            if (!resource.element.empty()) {
                add_enum(w, object, "element", view_of(resource.element));
            }
            break;
        }
        case ResourceType::BindlessArray: {
            add_uint(w, object, "slot_count", resource.slot_count);
            add_enum(w, object, "slot_type", view_of(resource.slot_type));
            break;
        }
        case ResourceType::Mesh: {
            add_string(w, object, "vertex_buffer", view_of(resource.vertex_buffer));
            add_string(w, object, "triangle_buffer", view_of(resource.triangle_buffer));
            break;
        }
        case ResourceType::ProceduralPrimitive: {
            add_string(w, object, "aabb_buffer", view_of(resource.aabb_buffer));
            break;
        }
        case ResourceType::Accel: {
            break;
        }
    }
    if (resource.input.kind != InputJson::Kind::None) {
        add_value(w, object, "input", make_input(w, resource.input));
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_resources(WriteCtx &w, const luisa::vector<ResourceJson> &resources) noexcept {
    auto *array = make_array(w, "resources");
    if (array == nullptr) { return nullptr; }
    for (auto &resource : resources) {
        auto *item = make_resource(w, resource);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("resources"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_shaders(WriteCtx &w, const luisa::vector<ShaderJson> &shaders) noexcept {
    auto *array = make_array(w, "shaders");
    if (array == nullptr) { return nullptr; }
    for (auto &shader : shaders) {
        auto *item = make_shader(w, shader);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("shaders"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_sampler(WriteCtx &w, const SamplerJson &sampler) noexcept {
    auto *object = make_object(w, "sampler");
    if (object == nullptr) { return nullptr; }
    add_enum(w, object, "filter", view_of(sampler.filter));
    add_enum(w, object, "address", view_of(sampler.address));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_binding(WriteCtx &w, const BindingJson &binding) noexcept {
    auto *object = make_object(w, "binding");
    if (object == nullptr) { return nullptr; }
    if (binding.has_index) {
        add_uint(w, object, "index", binding.index);
    } else {
        add_uint(w, object, "register", binding.reg);
        add_uint(w, object, "space", binding.space);
    }
    add_string(w, object, "resource", view_of(binding.resource));
    add_uint(w, object, "offset", binding.offset);
    add_uint(w, object, "size", binding.size);
    add_enum(w, object, "usage", view_of(binding.usage));
    return object;
}

[[nodiscard]] yyjson_mut_val *make_bindings(WriteCtx &w, const luisa::vector<BindingJson> &bindings) noexcept {
    auto *array = make_array(w, "bindings");
    if (array == nullptr) { return nullptr; }
    for (auto &binding : bindings) {
        auto *item = make_binding(w, binding);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("bindings"); }
    }
    return array;
}

[[nodiscard]] uint32_t read_u32_le(const std::byte *bytes) noexcept {
    auto result = uint32_t{0u};
    for (auto i = 0u; i < 4u; i++) {
        result |= static_cast<uint32_t>(std::to_integer<unsigned char>(bytes[i])) << (8u * i);
    }
    return result;
}

// A uniform payload is written back in the form that carries its type: the
// numeric spellings are re-encoded from their little-endian bytes, everything
// else (an explicit `hex`, or a payload whose size disagrees with its type)
// falls back to the hex form.
[[nodiscard]] yyjson_mut_val *make_uniform_payload(WriteCtx &w, luisa::string_view type,
                                                   const luisa::vector<std::byte> &bytes,
                                                   const char *what) noexcept {
    auto components = uint32_t{0u};
    auto numeric = uniform_type_layout(type, components) && components != 0u &&
                   bytes.size() == static_cast<size_t>(components) * 4u;
    if (!numeric) {
        auto *object = make_object(w, what);
        if (object == nullptr) { return nullptr; }
        auto hex = encode_hex(luisa::span<const std::byte>{bytes});
        add_string(w, object, "type", "hex");
        add_string(w, object, "hex", view_of(hex));
        return object;
    }
    auto *object = make_object(w, what);
    if (object == nullptr) { return nullptr; }
    add_string(w, object, "type", type);
    auto is_float = type.starts_with("float");
    auto is_unsigned = type.starts_with("uint");
    auto add_component = [&](yyjson_mut_val *target, uint32_t index) noexcept {
        auto bits = read_u32_le(bytes.data() + static_cast<size_t>(index) * 4u);
        if (is_float) {
            auto value = float{0.0f};
            std::memcpy(&value, &bits, sizeof(value));
            if (components == 1u) {
                add_real(w, target, "value", static_cast<double>(value));
            } else if (!yyjson_mut_arr_add_real(w.doc, target, static_cast<double>(value))) {
                w.fail("value");
            }
        } else if (is_unsigned) {
            if (components == 1u) {
                add_uint(w, target, "value", bits);
            } else if (!yyjson_mut_arr_add_uint(w.doc, target, bits)) {
                w.fail("value");
            }
        } else {
            auto value = static_cast<int32_t>(bits);
            if (components == 1u) {
                if (!yyjson_mut_obj_add_int(w.doc, target, "value", value)) { w.fail("value"); }
            } else if (!yyjson_mut_arr_add_int(w.doc, target, value)) {
                w.fail("value");
            }
        }
    };
    if (components == 1u) {
        add_component(object, 0u);
    } else {
        auto *value = make_array(w, "value");
        if (value == nullptr) { return object; }
        for (auto i = uint32_t{0u}; i < components; i++) { add_component(value, i); }
        add_value(w, object, "value", value);
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_uniform(WriteCtx &w, const UniformJson &uniform) noexcept {
    return make_uniform_payload(w, view_of(uniform.type), uniform.bytes, "uniform");
}

[[nodiscard]] yyjson_mut_val *make_uniforms(WriteCtx &w, const luisa::vector<UniformJson> &uniforms) noexcept {
    auto *array = make_array(w, "uniforms");
    if (array == nullptr) { return nullptr; }
    for (auto &uniform : uniforms) {
        auto *item = make_uniform(w, uniform);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("uniforms"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_argument(WriteCtx &w, const ArgumentJson &argument) noexcept {
    auto *object = make_object(w, "argument");
    if (object == nullptr) { return nullptr; }
    add_enum(w, object, "kind", view_of(argument.kind));
    if (argument.kind == "uniform") {
        // ArgumentJson keeps the bytes, not the type spelling, so the payload is
        // written in its self-describing hex form.
        auto hex = encode_hex(luisa::span<const std::byte>{argument.bytes});
        add_string(w, object, "hex", view_of(hex));
    } else {
        add_string(w, object, "resource", view_of(argument.resource));
        if (argument.kind == "texture") {
            add_uint(w, object, "level", argument.level);
        } else if (argument.kind == "buffer") {
            add_uint(w, object, "offset", argument.offset);
        }
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_arguments(WriteCtx &w, const luisa::vector<ArgumentJson> &arguments) noexcept {
    auto *array = make_array(w, "arguments");
    if (array == nullptr) { return nullptr; }
    for (auto &argument : arguments) {
        auto *item = make_argument(w, argument);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("arguments"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_modification(WriteCtx &w, const BindlessModJson &modification) noexcept {
    auto *object = make_object(w, "modification");
    if (object == nullptr) { return nullptr; }
    add_uint(w, object, "slot", modification.slot);
    add_enum(w, object, "kind", view_of(modification.kind));
    add_enum(w, object, "op", view_of(modification.op));
    add_string(w, object, "resource", view_of(modification.resource));
    add_uint(w, object, "offset", modification.offset);
    add_uint(w, object, "size", modification.size);
    if (modification.kind == "texture2d" || modification.kind == "texture3d") {
        add_value(w, object, "sampler", make_sampler(w, modification.sampler));
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_modifications(WriteCtx &w, const luisa::vector<BindlessModJson> &modifications) noexcept {
    auto *array = make_array(w, "modifications");
    if (array == nullptr) { return nullptr; }
    for (auto &modification : modifications) {
        auto *item = make_modification(w, modification);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("modifications"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_accel_modification(WriteCtx &w, const AccelModJson &modification) noexcept {
    auto *object = make_object(w, "accel modification");
    if (object == nullptr) { return nullptr; }
    add_uint(w, object, "index", modification.index);
    if (modification.has_user_id) { add_uint(w, object, "user_id", modification.user_id); }
    if (modification.has_visibility) { add_uint(w, object, "visibility", modification.visibility); }
    if (modification.has_opaque) { add_bool(w, object, "opaque", modification.opaque); }
    if (modification.has_transform) {
        add_value(w, object, "transform", make_real_array(w, modification.transform));
    }
    if (modification.has_primitive) {
        add_string(w, object, "primitive", view_of(modification.primitive));
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_accel_modifications(WriteCtx &w, const luisa::vector<AccelModJson> &modifications) noexcept {
    auto *array = make_array(w, "modifications");
    if (array == nullptr) { return nullptr; }
    for (auto &modification : modifications) {
        auto *item = make_accel_modification(w, modification);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("modifications"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_batched(WriteCtx &w, const luisa::vector<uint3> &batched) noexcept {
    auto *array = make_array(w, "batched");
    if (array == nullptr) { return nullptr; }
    for (auto value : batched) {
        auto *item = make_uint3(w, value);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("batched"); }
    }
    return array;
}

// `offset`/`size`/`src_offset`/`dst_offset` are written as the 3-element region
// form when the document carries one, and as the bare byte form otherwise.
void add_region(WriteCtx &w, yyjson_mut_val *object, const char *key,
                uint3 region, size_t bytes) noexcept {
    if (region.x != 0u || region.y != 0u || region.z != 0u) {
        add_value(w, object, key, make_uint3(w, region));
    } else {
        add_uint(w, object, key, bytes);
    }
}

[[nodiscard]] yyjson_mut_val *make_command(WriteCtx &w, const CommandJson &command) noexcept {
    auto *object = make_object(w, "command");
    if (object == nullptr) { return nullptr; }
    auto command_name = command_kind_name(command.kind);
    if (command_name == "unknown") {
        w.fail("the kind of a workflow entry");
    } else {
        add_string(w, object, "cmd", command_name);
    }
    switch (command.kind) {
        case CommandKind::BufferUpload: {
            add_string(w, object, "resource", view_of(command.resource));
            add_uint(w, object, "offset", command.offset);
            add_uint(w, object, "size", command.size);
            add_value(w, object, "input", make_input(w, command.input));
            break;
        }
        case CommandKind::BufferDownload: {
            add_string(w, object, "resource", view_of(command.resource));
            add_uint(w, object, "offset", command.offset);
            add_uint(w, object, "size", command.size);
            add_value(w, object, "output", make_output(w, command.output));
            add_value(w, object, "verify", make_verify(w, command.verify));
            break;
        }
        case CommandKind::BufferCopy: {
            add_string(w, object, "src", view_of(command.src));
            add_uint(w, object, "src_offset", command.src_offset);
            add_string(w, object, "dst", view_of(command.dst));
            add_uint(w, object, "dst_offset", command.dst_offset);
            add_uint(w, object, "size", command.size);
            break;
        }
        case CommandKind::TextureUpload: {
            add_string(w, object, "resource", view_of(command.resource));
            add_uint(w, object, "level", command.level);
            add_region(w, object, "offset", command.offset3, command.offset);
            add_region(w, object, "size", command.size3, command.size);
            add_enum(w, object, "storage", view_of(command.storage));
            add_value(w, object, "input", make_input(w, command.input));
            break;
        }
        case CommandKind::TextureDownload: {
            add_string(w, object, "resource", view_of(command.resource));
            add_uint(w, object, "level", command.level);
            add_region(w, object, "offset", command.offset3, command.offset);
            add_region(w, object, "size", command.size3, command.size);
            add_enum(w, object, "storage", view_of(command.storage));
            add_value(w, object, "output", make_output(w, command.output));
            break;
        }
        case CommandKind::TextureCopy: {
            add_enum(w, object, "storage", view_of(command.storage));
            add_string(w, object, "src", view_of(command.src));
            add_string(w, object, "dst", view_of(command.dst));
            add_uint(w, object, "src_level", command.src_level);
            add_uint(w, object, "dst_level", command.dst_level);
            add_region(w, object, "size", command.size3, command.size);
            add_region(w, object, "src_offset", command.src_offset3, command.src_offset);
            add_region(w, object, "dst_offset", command.dst_offset3, command.dst_offset);
            break;
        }
        case CommandKind::BufferToTextureCopy:
        case CommandKind::TextureToBufferCopy: {
            add_string(w, object, "buffer", view_of(command.buffer));
            add_uint(w, object, "buffer_offset", command.buffer_offset);
            add_string(w, object, "texture", view_of(command.texture));
            add_enum(w, object, "storage", view_of(command.storage));
            add_uint(w, object, "level", command.level);
            add_region(w, object, "size", command.size3, command.size);
            add_region(w, object, "offset", command.offset3, command.offset);
            break;
        }
        case CommandKind::NativeDispatch: {
            add_string(w, object, "shader", view_of(command.shader));
            add_value(w, object, "dispatch", make_uint3(w, command.dispatch));
            add_value(w, object, "grid", make_uint3(w, command.grid));
            add_value(w, object, "bindings", make_bindings(w, command.bindings));
            add_value(w, object, "uniforms", make_uniforms(w, command.uniforms));
            add_bool(w, object, "allow_usage_override", command.allow_usage_override);
            break;
        }
        case CommandKind::ShaderDispatch: {
            add_string(w, object, "shader", view_of(command.shader));
            add_value(w, object, "arguments", make_arguments(w, command.arguments));
            add_value(w, object, "dispatch", make_uint3(w, command.dispatch));
            // `indirect` is never written: no backend implements it, and the
            // parser rejects the key outright.
            add_value(w, object, "batched", make_batched(w, command.batched));
            break;
        }
        case CommandKind::BindlessArrayUpdate: {
            add_string(w, object, "resource", view_of(command.resource));
            add_enum(w, object, "mode", view_of(command.mode));
            add_value(w, object, "modifications", make_modifications(w, command.modifications));
            break;
        }
        case CommandKind::MeshBuild: {
            add_string(w, object, "resource", view_of(command.resource));
            add_enum(w, object, "request", view_of(command.request));
            add_string(w, object, "vertex_buffer", view_of(command.vertex_buffer));
            add_uint(w, object, "vertex_buffer_offset", command.vertex_buffer_offset);
            add_uint(w, object, "vertex_buffer_size", command.vertex_buffer_size);
            add_uint(w, object, "vertex_stride", command.vertex_stride);
            add_string(w, object, "triangle_buffer", view_of(command.triangle_buffer));
            add_uint(w, object, "triangle_buffer_offset", command.triangle_buffer_offset);
            add_uint(w, object, "triangle_buffer_size", command.triangle_buffer_size);
            break;
        }
        case CommandKind::ProceduralPrimitiveBuild: {
            add_string(w, object, "resource", view_of(command.resource));
            add_enum(w, object, "request", view_of(command.request));
            add_string(w, object, "aabb_buffer", view_of(command.aabb_buffer));
            add_uint(w, object, "aabb_buffer_offset", command.aabb_buffer_offset);
            add_uint(w, object, "aabb_buffer_size", command.aabb_buffer_size);
            break;
        }
        case CommandKind::AccelBuild: {
            add_string(w, object, "resource", view_of(command.resource));
            add_uint(w, object, "instance_count", command.instance_count);
            add_enum(w, object, "request", view_of(command.request));
            add_bool(w, object, "update_instance_buffer_only", command.update_instance_buffer_only);
            add_value(w, object, "modifications", make_accel_modifications(w, command.accel_modifications));
            break;
        }
        case CommandKind::CustomCommand: {
            add_uint(w, object, "uuid", command.uuid);
            add_string(w, object, "resource", view_of(command.resource));
            add_string(w, object, "buffer", view_of(command.buffer));
            add_uint(w, object, "buffer_offset", command.buffer_offset);
            add_string(w, object, "texture", view_of(command.texture));
            add_enum(w, object, "storage", view_of(command.storage));
            add_uint(w, object, "level", command.level);
            add_uint(w, object, "offset", command.offset);
            add_uint(w, object, "size", command.size);
            add_value(w, object, "input", make_input(w, command.input));
            add_value(w, object, "output", make_output(w, command.output));
            add_string(w, object, "label", view_of(command.label));
            if (command.uuid ==
                luisa::to_underlying(compute::CustomCommandUUID::NATIVE_SHADER_DISPATCH)) {
                add_string(w, object, "shader", view_of(command.shader));
                add_value(w, object, "dispatch", make_uint3(w, command.dispatch));
                add_value(w, object, "grid", make_uint3(w, command.grid));
                add_value(w, object, "bindings", make_bindings(w, command.bindings));
                add_value(w, object, "uniforms", make_uniforms(w, command.uniforms));
                add_bool(w, object, "allow_usage_override", command.allow_usage_override);
            }
            break;
        }
        case CommandKind::Log: {
            add_string(w, object, "message", view_of(command.message));
            break;
        }
        case CommandKind::Synchronize: {
            add_string(w, object, "label", view_of(command.label));
            break;
        }
    }
    return object;
}

[[nodiscard]] yyjson_mut_val *make_workflow(WriteCtx &w, const luisa::vector<CommandJson> &workflow) noexcept {
    auto *array = make_array(w, "workflow");
    if (array == nullptr) { return nullptr; }
    for (auto &command : workflow) {
        auto *item = make_command(w, command);
        if (item == nullptr) { break; }
        if (!yyjson_mut_arr_add_val(array, item)) { w.fail("workflow"); }
    }
    return array;
}

[[nodiscard]] yyjson_mut_val *make_root(WriteCtx &w, const DispatchJson &document) noexcept {
    auto *object = make_object(w, "document");
    if (object == nullptr) { return nullptr; }
    add_uint(w, object, "version", document.version);
    add_value(w, object, "mode", make_mode(w, document.mode));
    add_value(w, object, "config", make_config(w, document.config));
    add_value(w, object, "shaders", make_shaders(w, document.shaders));
    add_value(w, object, "resources", make_resources(w, document.resources));
    add_value(w, object, "workflow", make_workflow(w, document.workflow));
    return object;
}

// ---------------------------------------------------------------------------
// semantic validation
// ---------------------------------------------------------------------------

constexpr uint32_t kAnyResourceMask{0xffffffffu};

[[nodiscard]] constexpr uint32_t resource_mask(ResourceType type) noexcept {
    return 1u << luisa::to_underlying(type);
}

[[nodiscard]] constexpr uint32_t resource_mask(ResourceType a, ResourceType b) noexcept {
    return resource_mask(a) | resource_mask(b);
}

[[nodiscard]] constexpr uint32_t resource_mask(ResourceType a, ResourceType b, ResourceType c) noexcept {
    return resource_mask(a) | resource_mask(b) | resource_mask(c);
}

[[nodiscard]] luisa::string mask_names(uint32_t mask) noexcept {
    auto result = luisa::string{};
    for (auto spelling : kResourceTypeSpellings) {
        if ((mask & (1u << spelling.value)) == 0u) { continue; }
        if (!result.empty()) { result.append(" or "); }
        result.append(spelling.name);
    }
    return result.empty() ? luisa::string{"resource"} : result;
}

// Resource names are `[A-Za-z0-9_.-]{1,64}`.  An empty name is not an error: the
// caller fills in the sanitised file stem of the shader source.
[[nodiscard]] bool is_valid_resource_name(luisa::string_view name) noexcept {
    if (name.empty() || name.size() > 64u) { return false; }
    for (auto c : name) {
        auto valid = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                     (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        if (!valid) { return false; }
    }
    return true;
}

// The statically known byte size of a buffer resource, saturating on overflow.
[[nodiscard]] size_t buffer_bytes(const ResourceJson &resource) noexcept {
    if (!resource.element.empty()) {
        auto element_bytes = buffer_element_bytes(resource.element);
        if (element_bytes == 0u) { return 0u; }
        if (resource.count > std::numeric_limits<size_t>::max() / element_bytes) {
            return std::numeric_limits<size_t>::max();
        }
        return element_bytes * resource.count;
    }
    return resource.byte_size;
}

// The bytes one mip level of a texture/volume occupies, saturating on overflow.
[[nodiscard]] size_t mip_bytes(luisa::string_view storage, uint3 size, uint32_t level) noexcept {
    auto element_bytes = storage_bytes(storage);
    if (element_bytes == 0u) { return 0u; }
    auto extent = mip_extent(size, level);
    auto result = element_bytes;
    for (auto component : {extent.x, extent.y, extent.z}) {
        if (component == 0u) { return 0u; }
        if (result > std::numeric_limits<size_t>::max() / component) {
            return std::numeric_limits<size_t>::max();
        }
        result *= component;
    }
    return result;
}

// The bytes a resource holds (every mip level of a texture/volume), saturating on
// overflow; 0 for the kinds without a statically known size. The loop stops as
// soon as two levels occupy the same number of bytes (every further level is a
// 1x1x1 or a degenerate extent), so a document cannot make it spin on an absurd
// `levels` count.
[[nodiscard]] size_t resource_byte_size(const ResourceJson &resource) noexcept {
    constexpr auto max_size = std::numeric_limits<size_t>::max();
    switch (resource.type) {
        case ResourceType::Buffer: return buffer_bytes(resource);
        case ResourceType::Texture:
        case ResourceType::Volume: {
            auto total = size_t{0u};
            auto previous = size_t{0u};
            for (auto level = uint32_t{0u}; level < resource.levels; level++) {
                auto bytes = mip_bytes(view_of(resource.storage), resource.size, level);
                if (level != 0u && bytes == previous) {
                    auto remaining = static_cast<size_t>(resource.levels - level);
                    if (bytes != 0u && remaining > (max_size - total) / bytes) {
                        return max_size;
                    }
                    return total + bytes * remaining;
                }
                previous = bytes;
                if (bytes > max_size - total) { return max_size; }
                total += bytes;
            }
            return total;
        }
        default: return 0u;
    }
}

struct SemanticState {
    const DispatchJson &document;
    const JsonLimits &limits;
    Diagnostics &diag;
    luisa::unordered_map<luisa::string_view, const ResourceJson *> resources;
    // Every declared variant of a shader name: a name may be declared once per
    // language (see `declare_shader`).
    luisa::unordered_map<luisa::string_view, luisa::vector<const ShaderJson *>> shader_variants;
    luisa::unordered_map<luisa::string_view, const ShaderJson *> shaders;
    luisa::unordered_set<luisa::string_view> referenced_resources;
    luisa::unordered_set<luisa::string_view> referenced_shaders;

    void error(luisa::string message) noexcept { diag.error(std::move(message)); }
    void warning(luisa::string message) noexcept { diag.warning(std::move(message)); }

    // Records a declared shader. A name may be declared once per language - the
    // backend picks the variant it speaks - so a second declaration is only
    // rejected when the two cannot be told apart (either one derives its
    // language at run time, or they claim the same one).
    [[nodiscard]] bool declare_shader(luisa::string_view name,
                                      const ShaderJson *shader) noexcept {
        auto &variants = shader_variants[name];
        for (auto *other : variants) {
            if (!other->has_language || !shader->has_language ||
                other->language == shader->language) {
                return false;
            }
        }
        if (variants.empty()) { shaders.emplace(name, shader); }
        variants.emplace_back(shader);
        return true;
    }

    [[nodiscard]] const ResourceJson *resource(luisa::string_view name) noexcept {
        auto iter = resources.find(name);
        if (iter == resources.end()) { return nullptr; }
        referenced_resources.emplace(iter->first);
        return iter->second;
    }

    [[nodiscard]] const ResourceJson *resource(luisa::string_view name, bool mark_used) noexcept {
        auto iter = resources.find(name);
        if (iter == resources.end()) { return nullptr; }
        if (mark_used) { referenced_resources.emplace(iter->first); }
        return iter->second;
    }

    [[nodiscard]] const ShaderJson *shader(luisa::string_view name) noexcept {
        auto iter = shaders.find(name);
        if (iter == shaders.end()) { return nullptr; }
        referenced_shaders.emplace(iter->first);
        return iter->second;
    }

    // Every resource reference the workflow makes must name a declared entry of
    // the right kind; a dangling name is an error.
    [[nodiscard]] bool require_resource(luisa::string_view path, const char *field,
                                        luisa::string_view name, uint32_t mask) noexcept {
        if (name.empty()) {
            error(luisa::format(FMT_STRING("{}.{}: a {} resource name is required"),
                                path, field, mask_names(mask)));
            return false;
        }
        auto *found = resource(name);
        if (found == nullptr) {
            error(luisa::format(FMT_STRING("{}.{}: unknown resource '{}'"), path, field, name));
            return false;
        }
        if ((mask & resource_mask(found->type)) == 0u) {
            error(luisa::format(FMT_STRING("{}.{}: resource '{}' is a {}, expected {}"),
                                path, field, name, resource_type_name(found->type), mask_names(mask)));
            return false;
        }
        return true;
    }

    [[nodiscard]] bool require_shader(luisa::string_view path, const char *field,
                                      luisa::string_view name) noexcept {
        if (name.empty()) {
            error(luisa::format(FMT_STRING("{}.{}: a shader name is required"), path, field));
            return false;
        }
        if (shader(name) == nullptr) {
            error(luisa::format(FMT_STRING("{}.{}: unknown shader '{}'"), path, field, name));
            return false;
        }
        return true;
    }
};

void validate_output(SemanticState &state, luisa::string_view path, const OutputJson &output) noexcept {
    if (!output.discard && output.file.empty()) {
        state.error(luisa::format(FMT_STRING("{}.file: a non-discarding output needs a file"),
                                  path));
    }
}

void validate_input(SemanticState &state, luisa::string_view path, const InputJson &input,
                    bool require_payload, bool file_or_inline_only) noexcept {
    switch (input.kind) {
        case InputJson::Kind::None: {
            if (require_payload) {
                state.error(luisa::format(FMT_STRING("{}: a 'file' or 'inline' payload is required"), path));
            }
            break;
        }
        case InputJson::Kind::File: {
            if (input.file.empty()) {
                state.error(luisa::format(FMT_STRING("{}.file: empty file name"), path));
            }
            break;
        }
        case InputJson::Kind::Inline: {
            if (input.inline_bytes.empty()) {
                state.warning(luisa::format(FMT_STRING("{}: the inline payload is empty"), path));
            }
            break;
        }
        case InputJson::Kind::Resource: {
            if (file_or_inline_only) {
                state.error(luisa::format(FMT_STRING("{}: expected a 'file' or 'inline' payload, got a 'resource' one"),
                                          path));
            } else if (input.resource.empty()) {
                state.error(luisa::format(FMT_STRING("{}.resource: empty resource name"), path));
            } else {
                (void)state.require_resource(path, "resource", input.resource, kAnyResourceMask);
            }
            break;
        }
    }
}

// `offset`/`size` are byte regions of a buffer; a size of 0 means "rest".
void validate_buffer_region(SemanticState &state, luisa::string_view path,
                            luisa::string_view resource_name, size_t offset, size_t size) noexcept {
    auto *resource = state.resource(resource_name, false);
    if (resource == nullptr || resource->type != ResourceType::Buffer) { return; }
    auto total = buffer_bytes(*resource);
    if (total == 0u) { return; }// a zero-sized buffer is reported by its own check
    if (offset > total) {
        state.error(luisa::format(FMT_STRING("{}.offset: offset {} is beyond the {} bytes of buffer '{}'"),
                                  path, offset, total, resource_name));
        return;
    }
    if (size != 0u && size > total - offset) {
        state.error(luisa::format(FMT_STRING("{}.size: the region [{}, {}) exceeds the {} bytes of buffer '{}'"),
                                  path, offset, offset + size, total, resource_name));
    }
}

// The region of a texture command must live inside its mip level: the extent of
// level `level` is `max(1, extent >> level)` per component, and every component
// of `offset + size` is bounded by it.  A zero component means "rest of the
// level".  When the command names a storage it must agree with the resource.
void validate_texture_region(SemanticState &state, luisa::string_view path,
                             luisa::string_view resource_name, uint32_t level,
                             luisa::string_view storage, uint3 offset, uint3 size) noexcept {
    auto *resource = state.resource(resource_name, false);
    if (resource == nullptr) { return; }
    if (resource->type != ResourceType::Texture && resource->type != ResourceType::Volume) { return; }
    if (!storage.empty() && !resource->storage.empty() && storage != view_of(resource->storage)) {
        state.error(luisa::format(FMT_STRING("{}.storage: storage '{}' disagrees with resource '{}' ({})"),
                                  path, storage, resource_name, resource->storage));
    }
    if (resource->levels == 0u) { return; }// reported by the resource check
    if (level >= resource->levels) {
        state.error(luisa::format(FMT_STRING("{}.level: level {} is beyond the {} level(s) of resource '{}'"),
                                  path, level, resource->levels, resource_name));
        return;
    }
    auto extent = mip_extent(resource->size, level);
    auto end = uint3{offset.x + (size.x == 0u ? extent.x - (offset.x < extent.x ? offset.x : 0u) : size.x),
                     offset.y + (size.y == 0u ? extent.y - (offset.y < extent.y ? offset.y : 0u) : size.y),
                     offset.z + (size.z == 0u ? extent.z - (offset.z < extent.z ? offset.z : 0u) : size.z)};
    if (offset.x > extent.x || offset.y > extent.y || offset.z > extent.z ||
        end.x > extent.x || end.y > extent.y || end.z > extent.z) {
        state.error(luisa::format(FMT_STRING("{}: the region [{}, {}] x [{}, {}] x [{}, {}] is not inside the {}x{}x{} extent of mip {} of '{}'"),
                                  path, offset.x, end.x, offset.y, end.y, offset.z, end.z,
                                  extent.x, extent.y, extent.z, level, resource_name));
    }
}

// An inline texture upload must carry at least the tightly packed bytes of the
// addressed region - a smaller payload cannot fill it.  (A file source is
// checked against the file itself and may be pitch-padded, so it is exempt.)
void validate_inline_upload_size(SemanticState &state, luisa::string_view path,
                                 const CommandJson &command) noexcept {
    if (command.input.kind != InputJson::Kind::Inline) { return; }
    auto *resource = state.resource(command.resource, false);
    if (resource == nullptr || resource->levels == 0u) { return; }
    if (command.level >= resource->levels) { return; }
    auto storage = !command.storage.empty() ? view_of(command.storage) : view_of(resource->storage);
    auto element_bytes = storage_bytes(storage);
    if (element_bytes == 0u) { return; }
    auto extent = mip_extent(resource->size, command.level);
    auto region = uint3{command.size3.x == 0u ? extent.x : command.size3.x,
                        command.size3.y == 0u ? extent.y : command.size3.y,
                        command.size3.z == 0u ? extent.z : command.size3.z};
    auto expected = element_bytes;
    for (auto component : {region.x, region.y, region.z}) {
        if (component == 0u) { return; }
        if (expected > std::numeric_limits<size_t>::max() / component) { return; }
        expected *= component;
    }
    if (command.input.inline_bytes.size() < expected) {
        state.warning(luisa::format(FMT_STRING("{}.input: the inline payload of {} bytes is smaller than the {} bytes of the region"),
                                    path, command.input.inline_bytes.size(), expected));
    }
}

void validate_resource_entry(SemanticState &state, size_t index) noexcept {
    auto &resource = state.document.resources[index];
    auto path = path_index("resources", index);
    switch (resource.type) {
        case ResourceType::Buffer: {
            if (resource.element.empty()) {
                if (resource.byte_size == 0u) {
                    state.error(luisa::format(FMT_STRING("{}: buffer '{}' has no size (either 'element' with 'count' or 'byte_size' is required)"),
                                              path, resource.name));
                }
            } else {
                auto element_bytes = buffer_element_bytes(resource.element);
                if (resource.count == 0u) {
                    state.error(luisa::format(FMT_STRING("{}.count: buffer '{}' has a count of 0"),
                                              path, resource.name));
                }
                if (resource.has_byte_size) {
                    auto expected = element_bytes == 0u || resource.count > std::numeric_limits<size_t>::max() / element_bytes ?
                                        std::numeric_limits<size_t>::max() :
                                        element_bytes * resource.count;
                    if (expected != resource.byte_size) {
                        state.error(luisa::format(FMT_STRING("{}.byte_size: byte_size {} does not match element '{}' ({} bytes) x count {} = {}"),
                                                  path, resource.byte_size, resource.element,
                                                  element_bytes, resource.count, expected));
                    }
                }
            }
            break;
        }
        case ResourceType::Texture:
        case ResourceType::Volume: {
            if (resource.storage.empty()) {
                state.error(luisa::format(FMT_STRING("{}.storage: {} '{}' has no storage"),
                                          path, resource_type_name(resource.type), resource.name));
            }
            if (resource.levels == 0u) {
                state.error(luisa::format(FMT_STRING("{}.levels: {} '{}' needs at least one level"),
                                          path, resource_type_name(resource.type), resource.name));
            }
            if (resource.size.x == 0u || resource.size.y == 0u || resource.size.z == 0u) {
                state.error(luisa::format(FMT_STRING("{}.size: {} '{}' has a zero extent [x, y, z]"),
                                          path, resource_type_name(resource.type), resource.name));
            }
            if (resource.type == ResourceType::Texture && resource.size.z != 1u) {
                state.warning(luisa::format(FMT_STRING("{}.size: texture '{}' expects a z of 1, got {}"),
                                            path, resource.name, resource.size.z));
            }
            break;
        }
        case ResourceType::BindlessArray: {
            if (resource.slot_count == 0u) {
                state.error(luisa::format(FMT_STRING("{}.slot_count: bindless array '{}' has no slots"),
                                          path, resource.name));
            }
            break;
        }
        case ResourceType::Mesh: {
            (void)state.require_resource(path, "vertex_buffer", resource.vertex_buffer,
                                         resource_mask(ResourceType::Buffer));
            (void)state.require_resource(path, "triangle_buffer", resource.triangle_buffer,
                                         resource_mask(ResourceType::Buffer));
            break;
        }
        case ResourceType::ProceduralPrimitive: {
            // `aabb_buffer` is the optional creation-time AABB range: like a
            // mesh's vertex and triangle buffers it must name a `buffer`
            // resource, and one that holds `aabb` elements.
            if (!resource.aabb_buffer.empty()) {
                (void)state.require_resource(path, "aabb_buffer", resource.aabb_buffer,
                                             resource_mask(ResourceType::Buffer));
                auto *boxes = state.resource(resource.aabb_buffer, false);
                if (boxes != nullptr && boxes->type == ResourceType::Buffer &&
                    boxes->element != "aabb") {
                    state.error(luisa::format(FMT_STRING("{}.aabb_buffer: buffer '{}' must have the element 'aabb'"),
                                              path, resource.aabb_buffer));
                }
            }
            break;
        }
        case ResourceType::Accel: {
            break;
        }
    }
    // An absurd byte size never reaches the device: the allocator of a backend
    // aborts (or the process faults) on a resource no device can hold, which is a
    // crash rather than the diagnostic this budget turns it into.
    if (auto bytes = resource_byte_size(resource); bytes > state.limits.max_resource_bytes) {
        state.error(luisa::format(FMT_STRING("{}: {} '{}' is {} byte(s), which exceeds the limit of {} byte(s) (config.limits.max_resource_bytes)"),
                                  path, resource_type_name(resource.type), resource.name,
                                  bytes, state.limits.max_resource_bytes));
    }
    if (resource.input.kind != InputJson::Kind::None) {
        validate_input(state, path_key(path, "input"), resource.input, false, false);
    }
}

// Resources are created in dependency order; a cycle through the `resource`
// inputs cannot be ordered.
void validate_resource_input_cycles(SemanticState &state) noexcept {
    auto &resources = state.document.resources;
    auto count = resources.size();
    if (count == 0u) { return; }
    // 0 == unvisited, 1 == on the stack, 2 == done
    auto colors = luisa::vector<uint8_t>{};
    colors.resize(count, 0u);
    for (auto i = size_t{0u}; i < count; i++) {
        if (colors[i] != 0u) { continue; }
        auto index = i;
        auto guard = size_t{0u};
        while (colors[index] == 0u) {
            if (guard++ > count) { break; }
            colors[index] = 1u;
            auto &resource = resources[index];
            if (resource.input.kind != InputJson::Kind::Resource || resource.input.resource.empty()) { break; }
            auto iter = state.resources.find(view_of(resource.input.resource));
            if (iter == state.resources.end()) { break; }// dangling: reported elsewhere
            auto next = static_cast<size_t>(iter->second - resources.data());
            if (colors[next] == 1u) {
                state.error(luisa::format(FMT_STRING("resources[{}].input.resource: the input graph of '{}' is cyclic"),
                                          index, resource.name));
                break;
            }
            if (colors[next] == 2u) { break; }
            index = next;
        }
        for (auto &color : colors) {
            if (color == 1u) { color = 2u; }
        }
    }
}

void validate_native_dispatch(SemanticState &state, luisa::string_view path,
                              const CommandJson &command) noexcept {
    auto mask = resource_mask(ResourceType::Buffer, ResourceType::Texture, ResourceType::Volume);
    const ShaderJson *shader = nullptr;
    if (state.require_shader(path, "shader", command.shader)) {
        shader = state.shader(command.shader);
    }
    // `dispatch` and `grid` are alternatives and one of them must be non-zero.
    auto has_dispatch = command.dispatch.x != 0u || command.dispatch.y != 0u || command.dispatch.z != 0u;
    auto has_grid = command.grid.x != 0u || command.grid.y != 0u || command.grid.z != 0u;
    if (has_dispatch && has_grid) {
        state.error(luisa::format(FMT_STRING("{}: 'dispatch' and 'grid' are alternatives, not both"),
                                  path));
    } else if (!has_dispatch && !has_grid) {
        state.error(luisa::format(FMT_STRING("{}: either 'dispatch' or 'grid' must be non-zero"), path));
    }
    if (command.bindings.size() > state.limits.max_bindings_per_dispatch) {
        state.error(luisa::format(FMT_STRING("{}.bindings: {} bindings exceed the limit of {}"),
                                  path, command.bindings.size(), state.limits.max_bindings_per_dispatch));
    }
    auto seen_indices = luisa::unordered_set<uint32_t>{};
    auto seen_pairs = luisa::unordered_set<uint64_t>{};
    for (auto i = size_t{0u}; i < command.bindings.size(); i++) {
        auto &binding = command.bindings[i];
        auto binding_path = path_index(path_key(path, "bindings"), i);
        if (binding.has_index) {
            if (!seen_indices.emplace(binding.index).second) {
                state.error(luisa::format(FMT_STRING("{}.index: duplicate binding index {}"),
                                          binding_path, binding.index));
            }
        } else if (binding.has_register) {
            // Only an explicit register/space pair can collide: a binding with no
            // selector at all is positional, and the launcher fills the canonical
            // (space, register) order of the reflection table for it.
            auto pair = (static_cast<uint64_t>(binding.space) << 32u) | binding.reg;
            if (!seen_pairs.emplace(pair).second) {
                state.error(luisa::format(FMT_STRING("{}: duplicate binding register {} in space {}"),
                                          binding_path, binding.reg, binding.space));
            }
        }
        // A binding always names a resource: an empty name would otherwise reach
        // the launcher as the (path-less) message "no resource named ''".
        (void)state.require_resource(binding_path, "resource", binding.resource, mask);
        if (!binding.resource.empty()) {
            validate_buffer_region(state, binding_path, binding.resource,
                                   binding.offset, binding.size);
        }
        if (binding.usage == "none") {
            state.error(luisa::format(FMT_STRING("{}.usage: a binding needs a usage (read, write or read_write)"),
                                      binding_path));
        }
    }
    auto uniform_bytes = size_t{0u};
    for (auto &uniform : command.uniforms) {
        uniform_bytes += uniform.bytes.size();
    }
    if (uniform_bytes > state.limits.max_uniform_bytes) {
        state.error(luisa::format(FMT_STRING("{}.uniforms: {} uniform bytes exceed the limit of {}"),
                                  path, uniform_bytes, state.limits.max_uniform_bytes));
    }
    if (uniform_bytes != 0u && shader != nullptr) {
        auto push_constant_size = shader->push_constant_size != 0u ? shader->push_constant_size :
                                                                     (state.document.config.has_push_constant_size ?
                                                                          state.document.config.push_constant_size :
                                                                          0u);
        if (push_constant_size != 0u && uniform_bytes > push_constant_size) {
            state.error(luisa::format(FMT_STRING("{}.uniforms: {} uniform bytes exceed the push_constant_size ({}) of shader '{}'"),
                                      path, uniform_bytes, push_constant_size, command.shader));
        }
    }
    if (!command.uniforms.empty() && shader != nullptr && shader->push_constant_size == 0u) {
        state.warning(luisa::format(FMT_STRING("{}.uniforms: shader '{}' declares no push_constant_size; the payload is reflected"),
                                    path, command.shader));
    }
}

void validate_shader_dispatch(SemanticState &state, luisa::string_view path,
                              const CommandJson &command) noexcept {
    // A `shader_dispatch` either names a declared shader or one of the DSL
    // kernels the example registers itself.
    auto declared = state.shader(command.shader) != nullptr;
    auto builtin = false;
    for (auto name : builtin_dsl_kernels()) {
        if (name == command.shader) {
            builtin = true;
            break;
        }
    }
    if (!declared && !builtin) {
        state.error(luisa::format(FMT_STRING("{}.shader: unknown shader '{}' (not declared in "
                                             "`shaders` and not a registered DSL kernel)"),
                                  path, command.shader));
    }
    auto has_dispatch = command.dispatch.x != 0u || command.dispatch.y != 0u || command.dispatch.z != 0u;
    auto has_batched = !command.batched.empty();
    if (has_dispatch && has_batched) {
        state.error(luisa::format(FMT_STRING("{}: 'dispatch' and 'batched' are alternatives, not both"),
                                  path));
    } else if (!has_dispatch && !has_batched) {
        state.error(luisa::format(FMT_STRING("{}: a shader_dispatch needs 'dispatch' or 'batched'"),
                                  path));
    }
    if (has_dispatch && (command.dispatch.x == 0u || command.dispatch.y == 0u || command.dispatch.z == 0u)) {
        state.error(luisa::format(FMT_STRING("{}.dispatch: a dispatch of [0, 0, 0] threads does nothing"),
                                  path));
    }
    for (auto i = size_t{0u}; i < command.batched.size(); i++) {
        auto &entry = command.batched[i];
        if (entry.x == 0u || entry.y == 0u || entry.z == 0u) {
            state.error(luisa::format(FMT_STRING("{}[{}]: a batched dispatch of [0, 0, 0] does nothing"),
                                      path_key(path, "batched"), i));
        }
    }
    auto uniform_bytes = size_t{0u};
    for (auto i = size_t{0u}; i < command.arguments.size(); i++) {
        auto &argument = command.arguments[i];
        auto argument_path = path_index(path_key(path, "arguments"), i);
        if (argument.kind == "buffer") {
            (void)state.require_resource(argument_path, "resource", argument.resource,
                                         resource_mask(ResourceType::Buffer));
            validate_buffer_region(state, argument_path, argument.resource, argument.offset, 0u);
        } else if (argument.kind == "texture") {
            (void)state.require_resource(argument_path, "resource", argument.resource,
                                         resource_mask(ResourceType::Texture, ResourceType::Volume));
        } else if (argument.kind == "bindless_array") {
            (void)state.require_resource(argument_path, "resource", argument.resource,
                                         resource_mask(ResourceType::BindlessArray));
        } else if (argument.kind == "accel") {
            (void)state.require_resource(argument_path, "resource", argument.resource,
                                         resource_mask(ResourceType::Accel));
        } else if (argument.kind == "uniform") {
            uniform_bytes += argument.bytes.size();
        } else if (argument.kind.empty()) {
            state.error(luisa::format(FMT_STRING("{}: missing 'kind'"), argument_path));
        }
    }
    if (uniform_bytes > state.limits.max_uniform_bytes) {
        state.error(luisa::format(FMT_STRING("{}.arguments: {} uniform bytes exceed the limit of {}"),
                                  path, uniform_bytes, state.limits.max_uniform_bytes));
    }
}

void validate_bindless_update(SemanticState &state, luisa::string_view path,
                              const CommandJson &command) noexcept {
    auto *resource = state.resource(command.resource);
    if (command.resource.empty()) {
        state.error(luisa::format(FMT_STRING("{}.resource: a bindless array name is required"), path));
    } else if (resource == nullptr) {
        state.error(luisa::format(FMT_STRING("{}.resource: unknown resource '{}'"), path, command.resource));
    } else if (resource->type != ResourceType::BindlessArray) {
        state.error(luisa::format(FMT_STRING("{}.resource: resource '{}' is a {}, expected bindless_array"),
                                  path, command.resource, resource_type_name(resource->type)));
    }
    auto slot_count = resource != nullptr && resource->type == ResourceType::BindlessArray ?
                          resource->slot_count :
                          0u;
    auto slot_type = !command.mode.empty() ? luisa::string_view{command.mode} :
                     resource != nullptr   ? view_of(resource->slot_type) :
                                             luisa::string_view{};
    if (!command.mode.empty() && resource != nullptr && !resource->slot_type.empty() &&
        command.mode != view_of(resource->slot_type)) {
        state.warning(luisa::format(FMT_STRING("{}.mode: mode '{}' disagrees with the slot_type '{}' of resource '{}'"),
                                    path, command.mode, resource->slot_type, command.resource));
    }
    for (auto i = size_t{0u}; i < command.modifications.size(); i++) {
        auto &modification = command.modifications[i];
        auto modification_path = path_index(path_key(path, "modifications"), i);
        if (slot_count != 0u && modification.slot >= slot_count) {
            state.error(luisa::format(FMT_STRING("{}.slot: slot {} is beyond the {} slots of '{}'"),
                                      modification_path, modification.slot, slot_count, command.resource));
        }
        auto allowed = true;
        if (!slot_type.empty() && !modification.kind.empty()) {
            if (slot_type == "multiple") {
                allowed = true;
            } else {
                allowed = slot_type == modification.kind;
            }
            if (!allowed) {
                state.error(luisa::format(FMT_STRING("{}.kind: a '{}' modification is not valid for a '{}' array"),
                                          modification_path, modification.kind, slot_type));
            }
        }
        if (modification.op == "remove") {
            if (!modification.resource.empty() || modification.offset != 0u || modification.size != 0u) {
                state.warning(luisa::format(FMT_STRING("{}: a 'remove' modification ignores its payload"),
                                            modification_path));
            }
        } else if (modification.resource.empty()) {
            state.error(luisa::format(FMT_STRING("{}.resource: an '{}' modification needs a resource"),
                                      modification_path, modification.op));
        } else {
            auto mask = modification.kind == "buffer" ? resource_mask(ResourceType::Buffer) :
                        modification.kind.empty()     ? kAnyResourceMask :
                                                        resource_mask(ResourceType::Texture, ResourceType::Volume);
            (void)state.require_resource(modification_path, "resource", modification.resource, mask);
            if (modification.kind == "buffer") {
                validate_buffer_region(state, modification_path, modification.resource,
                                       modification.offset, modification.size);
            }
        }
    }
}

void validate_build(SemanticState &state, luisa::string_view path,
                    const CommandJson &command) noexcept {
    auto mask_buffer = resource_mask(ResourceType::Buffer);
    switch (command.kind) {
        case CommandKind::MeshBuild: {
            auto *resource = state.resource(command.resource);
            if (command.resource.empty() || resource == nullptr) {
                (void)state.require_resource(path, "resource", command.resource,
                                             resource_mask(ResourceType::Mesh));
            } else if (resource->type != ResourceType::Mesh) {
                state.error(luisa::format(FMT_STRING("{}.resource: resource '{}' is a {}, expected mesh"),
                                          path, command.resource, resource_type_name(resource->type)));
            }
            (void)state.require_resource(path, "vertex_buffer", command.vertex_buffer, mask_buffer);
            (void)state.require_resource(path, "triangle_buffer", command.triangle_buffer, mask_buffer);
            if (command.vertex_stride == 0u) {
                state.error(luisa::format(FMT_STRING("{}.vertex_stride: a mesh build needs a positive vertex_stride"),
                                          path));
            }
            if (command.triangle_buffer_size % 12u != 0u) {
                state.error(luisa::format(FMT_STRING("{}.triangle_buffer_size: {} is not a multiple of sizeof(Triangle) (12)"),
                                          path, command.triangle_buffer_size));
            }
            validate_buffer_region(state, path, command.vertex_buffer, command.vertex_buffer_offset,
                                   command.vertex_buffer_size);
            validate_buffer_region(state, path, command.triangle_buffer, command.triangle_buffer_offset,
                                   command.triangle_buffer_size);
            break;
        }
        case CommandKind::ProceduralPrimitiveBuild: {
            auto *resource = state.resource(command.resource);
            if (command.resource.empty() || resource == nullptr) {
                (void)state.require_resource(path, "resource", command.resource,
                                             resource_mask(ResourceType::ProceduralPrimitive));
            } else if (resource->type != ResourceType::ProceduralPrimitive) {
                state.error(luisa::format(FMT_STRING("{}.resource: resource '{}' is a {}, expected procedural_primitive"),
                                          path, command.resource, resource_type_name(resource->type)));
            }
            (void)state.require_resource(path, "aabb_buffer", command.aabb_buffer, mask_buffer);
            validate_buffer_region(state, path, command.aabb_buffer, command.aabb_buffer_offset,
                                   command.aabb_buffer_size);
            break;
        }
        case CommandKind::AccelBuild: {
            auto *resource = state.resource(command.resource);
            if (command.resource.empty() || resource == nullptr) {
                (void)state.require_resource(path, "resource", command.resource,
                                             resource_mask(ResourceType::Accel));
            } else if (resource->type != ResourceType::Accel) {
                state.error(luisa::format(FMT_STRING("{}.resource: resource '{}' is a {}, expected accel"),
                                          path, command.resource, resource_type_name(resource->type)));
            }
            if (command.update_instance_buffer_only && !command.accel_modifications.empty()) {
                state.error(luisa::format(FMT_STRING("{}: 'update_instance_buffer_only' is incompatible with non-empty 'modifications'"),
                                          path));
            }
            if (command.instance_count == 0u && command.accel_modifications.empty()) {
                state.error(luisa::format(FMT_STRING("{}.instance_count: an accel build without modifications needs a non-zero instance_count"),
                                          path));
            }
            for (auto i = size_t{0u}; i < command.accel_modifications.size(); i++) {
                auto &modification = command.accel_modifications[i];
                auto modification_path = path_index(path_key(path, "modifications"), i);
                if (command.instance_count != 0u && modification.index >= command.instance_count) {
                    state.error(luisa::format(FMT_STRING("{}.index: instance {} is beyond the {} instances of the build"),
                                              modification_path, modification.index, command.instance_count));
                }
                if (modification.has_primitive) {
                    (void)state.require_resource(modification_path, "primitive", modification.primitive,
                                                 resource_mask(ResourceType::Mesh,
                                                               ResourceType::ProceduralPrimitive));
                }
            }
            break;
        }
        case CommandKind::BufferUpload:
        case CommandKind::BufferDownload:
        case CommandKind::BufferCopy:
        case CommandKind::TextureUpload:
        case CommandKind::TextureDownload:
        case CommandKind::TextureCopy:
        case CommandKind::BufferToTextureCopy:
        case CommandKind::TextureToBufferCopy:
        case CommandKind::NativeDispatch:
        case CommandKind::ShaderDispatch:
        case CommandKind::BindlessArrayUpdate:
        case CommandKind::CustomCommand:
        case CommandKind::Log:
        case CommandKind::Synchronize: {
            // Every other kind is validated by its own pass; the list is
            // explicit so that a new enumerator is a compile-time reminder.
            break;
        }
    }
}

void validate_command(SemanticState &state, size_t index) noexcept {
    auto &command = state.document.workflow[index];
    auto path = path_index("workflow", index);
    auto mask_buffer = resource_mask(ResourceType::Buffer);
    auto mask_texture = resource_mask(ResourceType::Texture, ResourceType::Volume);
    switch (command.kind) {
        case CommandKind::BufferUpload: {
            (void)state.require_resource(path, "resource", command.resource, mask_buffer);
            validate_input(state, path_key(path, "input"), command.input, false, false);
            if (command.input.kind == InputJson::Kind::None) {
                state.warning(luisa::format(FMT_STRING("{}.input: an upload without a payload does nothing"),
                                            path));
            }
            validate_buffer_region(state, path, command.resource, command.offset, command.size);
            break;
        }
        case CommandKind::BufferDownload: {
            (void)state.require_resource(path, "resource", command.resource, mask_buffer);
            validate_output(state, path_key(path, "output"), command.output);
            validate_buffer_region(state, path, command.resource, command.offset, command.size);
            if (command.verify.kind != "none") {
                (void)state.require_resource(path_key(path, "verify"), "source",
                                             command.verify.source, mask_buffer);
            }
            break;
        }
        case CommandKind::BufferCopy: {
            (void)state.require_resource(path, "src", command.src, mask_buffer);
            (void)state.require_resource(path, "dst", command.dst, mask_buffer);
            validate_buffer_region(state, path, command.src, command.src_offset, command.size);
            validate_buffer_region(state, path, command.dst, command.dst_offset, command.size);
            if (command.size == 0u) {
                state.warning(luisa::format(FMT_STRING("{}.size: a size of 0 copies the rest of the source buffer"),
                                            path));
            }
            break;
        }
        case CommandKind::TextureUpload: {
            (void)state.require_resource(path, "resource", command.resource, mask_texture);
            validate_input(state, path_key(path, "input"), command.input, true, true);
            validate_texture_region(state, path, command.resource, command.level,
                                    view_of(command.storage), command.offset3, command.size3);
            validate_inline_upload_size(state, path, command);
            break;
        }
        case CommandKind::TextureDownload: {
            (void)state.require_resource(path, "resource", command.resource, mask_texture);
            validate_output(state, path_key(path, "output"), command.output);
            validate_texture_region(state, path, command.resource, command.level,
                                    view_of(command.storage), command.offset3, command.size3);
            break;
        }
        case CommandKind::TextureCopy: {
            (void)state.require_resource(path, "src", command.src, mask_texture);
            (void)state.require_resource(path, "dst", command.dst, mask_texture);
            validate_texture_region(state, path_key(path, "src"), command.src, command.src_level,
                                    view_of(command.storage), command.src_offset3, command.size3);
            validate_texture_region(state, path_key(path, "dst"), command.dst, command.dst_level,
                                    view_of(command.storage), command.dst_offset3, command.size3);
            break;
        }
        case CommandKind::BufferToTextureCopy:
        case CommandKind::TextureToBufferCopy: {
            (void)state.require_resource(path, "buffer", command.buffer, mask_buffer);
            (void)state.require_resource(path, "texture", command.texture, mask_texture);
            validate_texture_region(state, path, command.texture, command.level,
                                    view_of(command.storage), command.offset3, command.size3);
            validate_buffer_region(state, path, command.buffer, command.buffer_offset, 0u);
            break;
        }
        case CommandKind::NativeDispatch: {
            validate_native_dispatch(state, path, command);
            break;
        }
        case CommandKind::ShaderDispatch: {
            validate_shader_dispatch(state, path, command);
            break;
        }
        case CommandKind::BindlessArrayUpdate: {
            validate_bindless_update(state, path, command);
            break;
        }
        case CommandKind::MeshBuild:
        case CommandKind::ProceduralPrimitiveBuild:
        case CommandKind::AccelBuild: {
            validate_build(state, path, command);
            break;
        }
        case CommandKind::CustomCommand: {
            auto advertised = luisa::to_underlying(compute::CustomCommandUUID::NATIVE_SHADER_DISPATCH);
            auto dstorage = luisa::to_underlying(compute::CustomCommandUUID::DSTORAGE_READ);
            if (command.uuid != advertised && command.uuid != dstorage) {
                state.error(luisa::format(FMT_STRING("{}.uuid: unknown custom command 0x{:04x}; the registered commands are 0x{:04x} (native_shader_dispatch) and 0x{:04x} (dstorage_read)"),
                                          path, command.uuid, advertised, dstorage));
            }
            if (!command.resource.empty()) {
                (void)state.require_resource(path, "resource", command.resource, kAnyResourceMask);
            }
            if (!command.buffer.empty()) {
                (void)state.require_resource(path, "buffer", command.buffer, mask_buffer);
            }
            if (!command.texture.empty()) {
                (void)state.require_resource(path, "texture", command.texture, mask_texture);
            }
            if (command.input.kind != InputJson::Kind::None) {
                validate_input(state, path_key(path, "input"), command.input, false, false);
            }
            validate_output(state, path_key(path, "output"), command.output);
            // The `native_shader_dispatch` alias carries the whole native
            // dispatch field set, so it is validated exactly like
            // `native_dispatch`.
            if (command.uuid == advertised) {
                validate_native_dispatch(state, path, command);
            }
            break;
        }
        case CommandKind::Log:
        case CommandKind::Synchronize: {
            break;
        }
    }
}

void validate_mode(SemanticState &state) noexcept {
    auto &mode = state.document.mode;
    if (mode.frames < 1u) {
        state.error(luisa::format(FMT_STRING("mode.frames: an offline run needs at least one frame, got {}"),
                                  mode.frames));
    }
    // `exit_after_frames` is unsigned, so a negative value is rejected by the
    // parser (`expect_u32`) rather than here.
    if (!(mode.display_scale > 0.0f) || !std::isfinite(mode.display_scale)) {
        state.error(luisa::format(FMT_STRING("mode.display_scale: the exposure scale must be positive and finite, got {}"),
                                  mode.display_scale));
    }
    if (mode.window.width == 0u || mode.window.height == 0u) {
        state.error(luisa::format(FMT_STRING("mode.window: the window size must be non-zero, got {}x{}"),
                                  mode.window.width, mode.window.height));
    }
    if (mode.snapshot.every != 0u && mode.snapshot.path.empty()) {
        state.error(luisa::format(FMT_STRING("mode.snapshot.path: snapshotting every {} frames needs a path"),
                                  mode.snapshot.every));
    }
    if (mode.interactive) {
        if (mode.display_image.empty()) {
            state.error("mode.display_image: interactive mode needs a display image");
        } else {
            (void)state.require_resource("mode", "display_image", mode.display_image,
                                         resource_mask(ResourceType::Texture));
        }
        if (!mode.display_destination.empty() && mode.display_destination != "auto") {
            (void)state.require_resource("mode", "display_destination", mode.display_destination,
                                         resource_mask(ResourceType::Texture));
        }
        // The display pass reads the HDR image as float4 and writes the destination the
        // same way, so both ends have to be 4-channel float images.
        auto check_display_storage = [&](luisa::string_view which, luisa::string_view name) noexcept {
            auto *resource = state.resource(name, false);
            if (resource == nullptr) { return; }
            if (resource->storage == "float4" || resource->storage == "half4") { return; }
            state.error(luisa::format(FMT_STRING("mode.{}: '{}' uses storage '{}'; the display pass needs a 4-channel float image (float4 or half4)"),
                                      which, name, resource->storage));
        };
        if (!mode.display_image.empty()) { check_display_storage("display_image", mode.display_image); }
        if (!mode.display_destination.empty() && mode.display_destination != "auto") {
            check_display_storage("display_destination", mode.display_destination);
        }
    } else if (state.document.workflow.empty()) {
        state.warning("workflow: an offline run with an empty workflow does nothing");
    }
    if (mode.interactive && mode.dispatch_per_frame) {
        for (auto i = size_t{0u}; i < state.document.workflow.size(); i++) {
            auto kind = state.document.workflow[i].kind;
            if (kind == CommandKind::BufferDownload || kind == CommandKind::TextureDownload) {
                state.warning(luisa::format(FMT_STRING("workflow[{}]: 'dispatch_per_frame' re-runs the download every frame"),
                                            i));
                break;
            }
        }
    }
}

[[nodiscard]] bool validate_semantics(SemanticState &state) noexcept {
    auto &document = state.document;
    auto &limits = state.limits;
    if (document.resources.size() > limits.max_resources) {
        state.error(luisa::format(FMT_STRING("resources: {} entries exceed the limit of {}"),
                                  document.resources.size(), limits.max_resources));
    }
    if (document.shaders.size() > limits.max_shaders) {
        state.error(luisa::format(FMT_STRING("shaders: {} entries exceed the limit of {}"),
                                  document.shaders.size(), limits.max_shaders));
    }
    if (document.workflow.size() > limits.max_commands) {
        state.error(luisa::format(FMT_STRING("workflow: {} entries exceed the limit of {}"),
                                  document.workflow.size(), limits.max_commands));
    }
    for (auto i = size_t{0u}; i < document.resources.size(); i++) {
        auto &resource = document.resources[i];
        auto path = path_index("resources", i);
        if (!resource.name.empty()) {
            if (!is_valid_resource_name(resource.name)) {
                state.error(luisa::format(FMT_STRING("{}.name: invalid resource name '{}' (expected [A-Za-z0-9_.-]{{1,64}})"),
                                          path, resource.name));
            } else if (!state.resources.emplace(view_of(resource.name), &resource).second) {
                state.error(luisa::format(FMT_STRING("{}.name: duplicate resource name '{}'"),
                                          path, resource.name));
            }
        }
    }
    for (auto i = size_t{0u}; i < document.shaders.size(); i++) {
        auto &shader = document.shaders[i];
        auto path = path_index("shaders", i);
        if (!shader.name.empty()) {
            if (!is_valid_resource_name(shader.name)) {
                state.error(luisa::format(FMT_STRING("{}.name: invalid shader name '{}' (expected [A-Za-z0-9_.-]{{1,64}})"),
                                          path, shader.name));
            } else if (!state.declare_shader(view_of(shader.name), &shader)) {
                state.error(luisa::format(FMT_STRING("{}.name: duplicate shader name '{}' (a name may only be declared once per language)"),
                                          path, shader.name));
            }
        }
        if (shader.path.empty() && shader.source.empty()) {
            state.error(luisa::format(FMT_STRING("{}: shader '{}' needs either 'path' or 'source'"),
                                      path, shader.name));
        }
        // Shaders and resources live in separate namespaces; a collision is
        // legal but worth a warning.
        if (!shader.name.empty() && state.resources.find(view_of(shader.name)) != state.resources.end()) {
            state.warning(luisa::format(FMT_STRING("{}.name: shader '{}' shadows a resource of the same name (the two tables are separate namespaces)"),
                                        path, shader.name));
        }
    }
    for (auto i = size_t{0u}; i < document.resources.size(); i++) {
        validate_resource_entry(state, i);
    }
    validate_resource_input_cycles(state);
    for (auto i = size_t{0u}; i < document.workflow.size(); i++) {
        validate_command(state, i);
    }
    validate_mode(state);
    for (auto i = size_t{0u}; i < document.resources.size(); i++) {
        auto &resource = document.resources[i];
        if (resource.name.empty()) { continue; }
        if (state.referenced_resources.find(view_of(resource.name)) == state.referenced_resources.end()) {
            state.warning(luisa::format(FMT_STRING("resources[{}].name: resource '{}' is declared but never referenced"),
                                        i, resource.name));
        }
    }
    return !state.diag.has_error();
}

}// namespace

const char *to_string(ResourceType type) noexcept {
    return resource_type_name(type).data();
}

const char *to_string(CommandKind kind) noexcept {
    return command_kind_name(kind).data();
}

luisa::string_view command_kind_name(CommandKind kind) noexcept {
    for (auto spelling : kCommandKindSpellings) {
        if (spelling.value == luisa::to_underlying(kind)) { return spelling.name; }
    }
    return "unknown";
}

luisa::span<const luisa::string_view> builtin_dsl_kernels() noexcept {
    // Kept in sync with `register_builtin_kernels` in native_shader.cpp, which
    // is the only place that compiles them.
    static constexpr luisa::string_view kernels[]{
        "hdr_to_display",
        "fill_hdr_gradient",
        "scale_buffer",
    };
    return kernels;
}

luisa::span<const luisa::string_view> unsupported_command_kinds() noexcept {
    // A retired command kind is not a typo: no backend implements curve or
    // motion-blur acceleration structures, so the codec recognises the name and
    // says *why* it refuses it instead of listing the kinds it does know.
    static constexpr luisa::string_view kinds[]{
        "curve_build",
        "motion_instance_build",
    };
    return kinds;
}

luisa::span<const luisa::string_view> unsupported_resource_types() noexcept {
    // The same for the resource kinds: `curve` and `motion_instance` need the
    // acceleration structures above, and `indirect_dispatch_buffer` exists only
    // to feed an indirect dispatch, which this example rejects everywhere.
    static constexpr luisa::string_view types[]{
        "curve",
        "motion_instance",
        "indirect_dispatch_buffer",
    };
    return types;
}

namespace {
// The names of a spelling table, without its numeric payload: the runtime only
// needs the strings, and `Spelling` is file-local.
template<size_t N>
[[nodiscard]] luisa::span<const luisa::string_view> spelling_names(const Spelling (&table)[N]) noexcept {
    static_assert(N > 0u);
    static thread_local luisa::vector<luisa::string_view> names;
    if (names.size() != N) {
        names.clear();
        names.reserve(N);
        for (auto &&spelling : table) { names.emplace_back(spelling.name); }
    }
    return luisa::span{names};
}
}// namespace

luisa::span<const luisa::string_view> buffer_element_spellings() noexcept {
    return spelling_names(kBufferElementSpellings);
}

luisa::span<const luisa::string_view> pixel_storage_spellings() noexcept {
    return spelling_names(kStorageSpellings);
}

luisa::span<const luisa::string_view> usage_spellings() noexcept {
    return spelling_names(kUsageSpellings);
}

luisa::span<const luisa::string_view> command_kind_spellings() noexcept {
    return spelling_names(kCommandKindSpellings);
}

luisa::span<const luisa::string_view> resource_type_spellings() noexcept {
    return spelling_names(kResourceTypeSpellings);
}

bool parse_command_kind(luisa::string_view name, CommandKind &kind) noexcept {
    auto *spelling = find_spelling(kCommandKindSpellings, canonical_spelling(name));
    if (spelling == nullptr) { return false; }
    kind = static_cast<CommandKind>(spelling->value);
    return true;
}

luisa::string_view resource_type_name(ResourceType type) noexcept {
    for (auto spelling : kResourceTypeSpellings) {
        if (spelling.value == luisa::to_underlying(type)) { return spelling.name; }
    }
    return "unknown";
}

bool parse_resource_type(luisa::string_view name, ResourceType &type) noexcept {
    auto *spelling = find_spelling(kResourceTypeSpellings, canonical_spelling(name));
    if (spelling == nullptr) { return false; }
    type = static_cast<ResourceType>(spelling->value);
    return true;
}

bool decode_hex(luisa::string_view text, size_t max_bytes,
                luisa::vector<std::byte> &out, luisa::string &error) noexcept {
    out.clear();
    error.clear();
    if (text.size() % 2u != 0u) {
        error = luisa::format(FMT_STRING("hex payload must have an even number of digits, got {}"),
                              text.size());
        return false;
    }
    auto byte_count = text.size() / 2u;
    if (byte_count > max_bytes) {
        error = luisa::format(FMT_STRING("hex payload of {} bytes exceeds the limit of {} bytes"),
                              byte_count, max_bytes);
        return false;
    }
    out.reserve(byte_count);
    for (auto i = size_t{0u}; i < text.size(); i += 2u) {
        auto high = hex_nibble(text[i]);
        auto low = hex_nibble(text[i + 1u]);
        if (high < 0 || low < 0) {
            error = luisa::format(FMT_STRING("invalid hex digit at offset {}"), i);
            return false;
        }
        out.emplace_back(static_cast<std::byte>(
            static_cast<unsigned char>((high << 4) | low)));
    }
    return true;
}

luisa::string encode_hex(luisa::span<const std::byte> bytes) noexcept {
    constexpr char kDigits[]{"0123456789abcdef"};
    auto result = luisa::string{};
    result.reserve(bytes.size() * 2u);
    for (auto byte : bytes) {
        auto value = static_cast<uint32_t>(std::to_integer<unsigned char>(byte));
        result.push_back(kDigits[value >> 4u]);
        result.push_back(kDigits[value & 0x0fu]);
    }
    return result;
}

ParseResult parse_dispatch_json(luisa::string_view text, const JsonLimits &limits) noexcept {
    auto diagnostics = Diagnostics{limits};
    auto result = ParseResult{};
    auto finish = [&]() noexcept {
        result.errors = diagnostics.errors();
        result.warnings = diagnostics.warnings();
        if (diagnostics.has_error()) { result.value = luisa::nullopt; }
        return std::move(result);
    };
    if (text.size() > limits.max_document_bytes) {
        diagnostics.error(luisa::format(FMT_STRING("document of {} bytes exceeds the limit of {} bytes"),
                                        text.size(), limits.max_document_bytes));
        return finish();
    }
    // `yyjson_read_opts` copies the input (no YYJSON_READ_INSITU), so the
    // const_cast is legal and the document outlives `text`.
    auto read_error = yyjson_read_err{};
    auto *doc = yyjson_read_opts(const_cast<char *>(text.data()), text.size(),
                                 YYJSON_READ_ALLOW_BOM, nullptr, &read_error);
    if (doc == nullptr) {
        // 1-based line/column of the byte offset the reader stopped at.
        auto line = size_t{0u};
        auto column = size_t{0u};
        auto character = size_t{0u};
        yyjson_locate_pos(text.data(), text.size(), read_error.pos, &line, &column, &character);
        diagnostics.error(luisa::format(FMT_STRING("line {} column {}: {}"),
                                        line, column, read_error.msg));
        return finish();
    }
    auto *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        diagnostics.error(luisa::format(FMT_STRING("expected the document root to be an object, got {}"),
                                        json_type_name(root)));
    } else {
        auto ctx = Ctx{diagnostics, limits, 0u};
        result.value = parse_root(ctx, root);
    }
    yyjson_doc_free(doc);
    return finish();
}

ParseResult parse_dispatch_file(const luisa::filesystem::path &path,
                                const JsonLimits &limits) noexcept {
    auto failure = [](luisa::string message) noexcept {
        auto result = ParseResult{};
        result.errors.emplace_back(std::move(message));
        return result;
    };
    auto path_text = luisa::to_string(path);
    auto error_code = std::error_code{};
    if (!luisa::filesystem::exists(path, error_code)) {
        return failure(luisa::format(FMT_STRING("{}: no such file"), path_text));
    }
    if (luisa::filesystem::is_directory(path, error_code)) {
        return failure(luisa::format(FMT_STRING("{}: is a directory, expected a dispatch document"),
                                     path_text));
    }
    auto size = luisa::filesystem::file_size(path, error_code);
    if (error_code) {
        return failure(luisa::format(FMT_STRING("{}: cannot determine the file size"), path_text));
    }
    if (size == 0u) {
        return failure(luisa::format(FMT_STRING("{}: file is empty"), path_text));
    }
    if (size > limits.max_document_bytes) {
        return failure(luisa::format(FMT_STRING("{}: file of {} bytes exceeds the limit of {} bytes"),
                                     path_text, size, limits.max_document_bytes));
    }
    auto stream = std::ifstream{path, std::ios::binary};
    if (!stream.is_open()) {
        return failure(luisa::format(FMT_STRING("{}: cannot open the file for reading"), path_text));
    }
    auto text = luisa::string{};
    text.resize(size, '\0');
    stream.read(text.data(), static_cast<std::streamsize>(size));
    if (!stream) {
        return failure(luisa::format(FMT_STRING("{}: failed to read {} bytes"), path_text, size));
    }
    stream.close();
    return parse_dispatch_json(luisa::string_view{text.data(), text.size()}, limits);
}

WriteResult write_dispatch_json(const DispatchJson &document, const WriteOptions &options) noexcept {
    auto result = WriteResult{};
    auto *doc = yyjson_mut_doc_new(nullptr);
    if (doc == nullptr) {
        result.error = "failed to allocate a JSON document (out of memory?)";
        return result;
    }
    auto writer = WriteCtx{doc, {}};
    if (auto *root = make_root(writer, document)) {
        yyjson_mut_doc_set_root(doc, root);
    }
    if (writer.error.empty()) {
        auto length = size_t{0u};
        auto write_error = yyjson_write_err{};
        auto *json = yyjson_mut_write_opts(
            doc, options.pretty ? YYJSON_WRITE_PRETTY_TWO_SPACES : YYJSON_WRITE_NOFLAG,
            nullptr, &length, &write_error);
        if (json == nullptr) {
            result.error = luisa::format(FMT_STRING("failed to serialise the document: {}"),
                                         write_error.msg);
        } else {
            result.json = luisa::string{json, length};
            // The writer allocated the string with the default (libc) allocator.
            free(json);
        }
    } else {
        result.error = std::move(writer.error);
    }
    yyjson_mut_doc_free(doc);
    return result;
}

bool validate_dispatch_semantics(const DispatchJson &document, const JsonLimits &limits,
                                 luisa::vector<luisa::string> &errors,
                                 luisa::vector<luisa::string> &warnings) noexcept {
    // The semantic pass re-runs over the parse diagnostics, so it seeds its
    // warning filter with what the caller already collected.
    auto diag = Diagnostics{limits, warnings};
    auto state = SemanticState{document, limits, diag, {}, {}, {}, {}};
    (void)validate_semantics(state);
    for (auto &error : diag.errors()) { errors.emplace_back(error); }
    for (auto &warning : diag.warnings()) { warnings.emplace_back(warning); }
    return errors.empty();
}

bool equivalent(const DispatchJson &a, const DispatchJson &b, luisa::string &mismatch) noexcept {
    mismatch.clear();
    auto options = WriteOptions{.pretty = false};
    auto first = write_dispatch_json(a, options);
    auto second = write_dispatch_json(b, options);
    if (!first.error.empty()) {
        mismatch = luisa::format(FMT_STRING("failed to serialise the first document: {}"), first.error);
        return false;
    }
    if (!second.error.empty()) {
        mismatch = luisa::format(FMT_STRING("failed to serialise the second document: {}"), second.error);
        return false;
    }
    if (first.json == second.json) { return true; }
    auto limit = first.json.size() < second.json.size() ? first.json.size() : second.json.size();
    auto offset = size_t{0u};
    while (offset < limit && first.json[offset] == second.json[offset]) { offset++; }
    auto line = size_t{1u};
    auto column = size_t{1u};
    for (auto i = size_t{0u}; i < offset; i++) {
        if (first.json[i] == '\n') {
            line++;
            column = 1u;
        } else {
            column++;
        }
    }
    auto excerpt = [offset](const luisa::string &text) noexcept {
        auto begin = offset > 24u ? offset - 24u : size_t{0u};
        auto end = text.size() < offset + 24u ? text.size() : offset + 24u;
        return luisa::string{text.data() + begin, end - begin};
    };
    mismatch = luisa::format(FMT_STRING("canonical forms differ at line {} column {} (offset {}): '{}' != '{}'"),
                             line, column, offset, excerpt(first.json), excerpt(second.json));
    return false;
}

}// namespace luisa::native_shader
