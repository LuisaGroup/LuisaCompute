// Runtime side of the native shader example: resource registry, shader
// registry, workflow executor and sinks.
//
// This translation unit owns *device* work only: creating the resources a
// dispatch document asks for (including dstorage input loading), turning the
// document's command list into a `CommandList`, downloading results and
// verifying them on the host. It never parses JSON (native_shader_dispatch
// does) and never decides what to run (native_shader.cpp does).
//
// Ownership model: the runtime wrappers (`Buffer<T>`, `Image<T>`, `Accel`, ...)
// are templated on types that are data-dependent, so the registries own them
// through a type-erased `Owner` and keep the flat metadata the rest of the
// example needs (device handle, native handle, byte size, extent, storage).
// Typed views are reconstructed on demand from that metadata - this is the one
// place where type erasure is genuinely required, and it keeps the buffer
// element table (15 element types) free of a 15-way variant.
#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/backends/ext/native_shader_ext.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/procedural_primitive.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>

#include "native_shader_dispatch.h"

namespace luisa::native_shader {

using compute::Accel;
using compute::AccelBuildRequest;
using compute::BindlessArray;
using compute::BindlessSlotType;
using compute::Buffer;
using compute::BufferView;
using compute::ByteBuffer;
using compute::Device;
using compute::Image;
using compute::invalid_resource_handle;
using compute::Mesh;
using compute::NativeShader;
using compute::NativeShaderExt;
using compute::NativeShaderResourceBinding;
using compute::PixelStorage;
using compute::ProceduralPrimitive;
using compute::Sampler;
using compute::Stream;
using compute::TimelineEvent;
using compute::Usage;

// ---------------------------------------------------------------------------
// element / storage / enum tables (single source of truth for the JSON names)
// ---------------------------------------------------------------------------

// Buffer element types the document can ask for. `Triangle` and `Aabb` exist
// because the ray-tracing resources are created from typed buffers.
enum class BufferElement : uint32_t {
    Float,
    Float2,
    Float3,
    Float4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Int,
    Int2,
    Int3,
    Int4,
    Byte,
    Triangle,
    Aabb,
};

[[nodiscard]] size_t buffer_element_size(BufferElement element) noexcept;
[[nodiscard]] luisa::string_view buffer_element_name(BufferElement element) noexcept;
[[nodiscard]] bool parse_buffer_element(luisa::string_view name, BufferElement &element) noexcept;

[[nodiscard]] luisa::string_view pixel_storage_name(PixelStorage storage) noexcept;
[[nodiscard]] bool parse_pixel_storage(luisa::string_view name, PixelStorage &storage) noexcept;
[[nodiscard]] bool parse_usage(luisa::string_view name, Usage &usage) noexcept;
[[nodiscard]] luisa::string_view usage_name(Usage usage) noexcept;
[[nodiscard]] bool parse_slot_type(luisa::string_view name, BindlessSlotType &type) noexcept;
[[nodiscard]] bool parse_accel_request(luisa::string_view name, AccelBuildRequest &request) noexcept;
[[nodiscard]] bool parse_sampler(const SamplerJson &json, Sampler &sampler) noexcept;
[[nodiscard]] luisa::string_view native_shader_language_name(NativeShaderLanguage language) noexcept;
[[nodiscard]] bool parse_native_shader_language(luisa::string_view name,
                                                NativeShaderLanguage &language) noexcept;

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

// Error/warning sink shared by the registries and the executor. Errors are
// plain strings ("<path>: <message>" or "<name>: <message>") so that the first
// failure can be reported without aborting the process.
struct Diagnostics {
    luisa::vector<luisa::string> errors;
    luisa::vector<luisa::string> warnings;
    size_t max_errors{32u};
    void error(luisa::string message);
    void warning(luisa::string message);
    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

// Resolves the document's relative paths: input paths against the document
// directory (or `--workdir`), output paths additionally prefixed with the
// output directory. Never builds a `std::filesystem::path` from raw UTF-8
// (that conversion throws on undecodable bytes) - `path_from_narrow` is used.
class PathResolver {
private:
    luisa::filesystem::path _document_dir;
    luisa::filesystem::path _workdir;
    luisa::filesystem::path _output_dir;
    bool _has_workdir{false};
    Diagnostics *_diagnostics{nullptr};

public:
    PathResolver() noexcept = default;
    void set_diagnostics(Diagnostics &diagnostics) noexcept { _diagnostics = &diagnostics; }
    void set_document_dir(luisa::string_view directory) noexcept;
    void set_workdir(luisa::string_view directory) noexcept;
    void set_output_dir(luisa::string_view directory) noexcept;
    [[nodiscard]] const luisa::filesystem::path &document_dir() const noexcept { return _document_dir; }
    [[nodiscard]] const luisa::filesystem::path &output_dir() const noexcept { return _output_dir; }
    [[nodiscard]] luisa::filesystem::path resolve_input(luisa::string_view path) const noexcept;
    [[nodiscard]] luisa::filesystem::path resolve_output(luisa::string_view path) const noexcept;
    [[nodiscard]] luisa::filesystem::path resolve_shader(luisa::string_view path) const noexcept;
    // Creates the parent directory of `path`; false + message on failure.
    [[nodiscard]] bool ensure_parent_directory(const luisa::filesystem::path &path,
                                               luisa::string &error) noexcept;
};

// ---------------------------------------------------------------------------
// type-erased ownership
// ---------------------------------------------------------------------------

// Owns a runtime resource object of an unknown concrete type. The destructor is
// captured at creation time, so the registries never need a variant over the
// element types.
class Owner {
private:
    void *_object{nullptr};
    void (*_destroy)(void *) noexcept {nullptr};

public:
    Owner() noexcept = default;
    template<typename T>
    [[nodiscard]] static Owner create(T &&object) noexcept {
        using U = std::remove_cvref_t<T>;
        Owner owner;
        owner._object = new U{std::forward<T>(object)};
        owner._destroy = [](void *p) noexcept { delete static_cast<U *>(p); };
        return owner;
    }
    Owner(Owner &&other) noexcept : _object{other._object}, _destroy{other._destroy} {
        other._object = nullptr;
        other._destroy = nullptr;
    }
    Owner &operator=(Owner &&other) noexcept {
        if (this != &other) {
            reset();
            _object = other._object;
            _destroy = other._destroy;
            other._object = nullptr;
            other._destroy = nullptr;
        }
        return *this;
    }
    Owner(const Owner &) = delete;
    Owner &operator=(const Owner &) = delete;
    ~Owner() noexcept { reset(); }
    void reset() noexcept {
        if (_object != nullptr && _destroy != nullptr) { _destroy(_object); }
        _object = nullptr;
        _destroy = nullptr;
    }
    [[nodiscard]] bool valid() const noexcept { return _object != nullptr; }
};

// One created device resource plus the flat metadata the example uses.
struct OwnedResource {
    Owner owner;
    uint64_t handle{invalid_resource_handle};
    void *native_handle{nullptr};
    size_t stride{1u};   // buffer element stride in bytes
    size_t byte_size{0u};// buffer byte size / full image or volume size
    uint3 extent{0u, 0u, 0u};
    PixelStorage storage{PixelStorage::BYTE1};
    uint32_t levels{1u};
    // Non-null only for an `Image<float>`, which the ImGui window binds by
    // reference (`ImGuiWindow::register_texture(const Image<float> &, ...)`).
    Image<float> *float_image{nullptr};
};

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

class ResourceRegistry {
public:
    struct Entry {
        ResourceJson spec;
        OwnedResource resource;
    };

private:
    luisa::vector<Entry> _entries;
    luisa::unordered_map<luisa::string, size_t> _index;

public:
    // Creates every resource in dependency order and loads its `input`.
    // `dstorage_stream` may be null (the host fallback is used). False +
    // diagnostics on failure.
    [[nodiscard]] bool create_all(Device &device,
                                  Stream &stream,
                                  Stream *dstorage_stream,
                                  const DispatchJson &document,
                                  const PathResolver &paths,
                                  Diagnostics &diagnostics) noexcept;

    [[nodiscard]] const Entry *find(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::span<const Entry> entries() const noexcept { return _entries; }
    [[nodiscard]] size_t size() const noexcept { return _entries.size(); }

    // Typed view over an existing buffer resource (used to create the mesh and
    // procedural-primitive resources and to bind native dispatch uniforms).
    template<typename T>
    [[nodiscard]] BufferView<T> view_as(const Entry &entry,
                                        size_t offset_bytes = 0u,
                                        size_t byte_size = 0u) const noexcept {
        auto bytes = byte_size == 0u ? entry.resource.byte_size - offset_bytes : byte_size;
        return BufferView<T>{entry.resource.native_handle, entry.resource.handle,
                             sizeof(T), offset_bytes, bytes / sizeof(T),
                             entry.resource.byte_size / sizeof(T)};
    }
};

// ---------------------------------------------------------------------------
// shaders
// ---------------------------------------------------------------------------

// A compiled native shader plus its metadata, or a compiled DSL shader plus the
// encoder parameters the generic `shader_dispatch` path needs.
class ShaderRegistry {
public:
    struct NativeEntry {
        luisa::string name;
        NativeShader shader;
        luisa::string source_path;// provenance for logging (may be empty)
    };
    struct DslEntry {
        luisa::string name;
        Owner owner;
        uint64_t handle{invalid_resource_handle};
        size_t argument_count{0u};
        size_t uniform_size{0u};
        uint3 block_size{0u, 0u, 0u};
        uint32_t dimension{1u};
    };

private:
    Device *_device{nullptr};
    NativeShaderExt *_ext{nullptr};
    luisa::vector<NativeEntry> _native;
    luisa::vector<DslEntry> _dsl;
    luisa::string _dstorage_note;

public:
    // `ext` may be null: native shaders are then skipped with a warning (the
    // document's DSL-only parts still run).
    void set_device(Device &device, NativeShaderExt *ext) noexcept {
        _device = &device;
        _ext = ext;
    }
    [[nodiscard]] NativeShaderExt *extension() const noexcept { return _ext; }
    [[nodiscard]] bool has_native_support() const noexcept { return _ext != nullptr; }

    // Merges the document's shaders with the CLI-provided ones (CLI wins),
    // validates the language/backend matrix and compiles every native shader.
    [[nodiscard]] bool compile_all(const DispatchJson &document,
                                   luisa::span<const ShaderJson> cli_shaders,
                                   luisa::string_view backend,
                                   NativeShaderExt *ext,
                                   const PathResolver &paths,
                                   Diagnostics &diagnostics) noexcept;

    // Registers an already compiled DSL kernel (the display kernel, the
    // `shader_dispatch` corpus kernels).
    void add_dsl(luisa::string name,
                 Owner owner,
                 uint64_t handle,
                 size_t argument_count,
                 size_t uniform_size,
                 uint3 block_size,
                 uint32_t dimension) noexcept;

    [[nodiscard]] const NativeEntry *find_native(luisa::string_view name) const noexcept;
    [[nodiscard]] const DslEntry *find_dsl(luisa::string_view name) const noexcept;
    [[nodiscard]] bool has_shader(luisa::string_view name) const noexcept;
    [[nodiscard]] luisa::span<const NativeEntry> native_shaders() const noexcept { return _native; }
    [[nodiscard]] luisa::span<const DslEntry> dsl_shaders() const noexcept { return _dsl; }
};

// ---------------------------------------------------------------------------
// workflow execution
// ---------------------------------------------------------------------------

class WorkflowExecutor {
private:
    Device &_device;
    Stream &_stream;
    Stream *_upload_stream{nullptr};// dstorage stream, when available
    ResourceRegistry &_resources;
    ShaderRegistry &_shaders;
    const PathResolver &_paths;
    Diagnostics &_diagnostics;
    bool _write_sinks{true};
    bool _sync_uploads{false};

public:
    WorkflowExecutor(Device &device,
                     Stream &stream,
                     Stream *upload_stream,
                     ResourceRegistry &resources,
                     ShaderRegistry &shaders,
                     const PathResolver &paths,
                     Diagnostics &diagnostics) noexcept
        : _device{device}, _stream{stream}, _upload_stream{upload_stream},
          _resources{resources}, _shaders{shaders}, _paths{paths},
          _diagnostics{diagnostics} {}
    void set_write_sinks(bool write) noexcept { _write_sinks = write; }
    void set_sync_uploads(bool sync) noexcept { _sync_uploads = sync; }

    // Builds and submits the whole workflow once. When `write_sinks` is on,
    // downloads are written to their files (and verified) before returning, so
    // the call is a complete, self-checking execution of one frame.
    [[nodiscard]] bool execute(const DispatchJson &document) noexcept;

    // The number of device commands the last `execute` submitted.
    [[nodiscard]] size_t last_command_count() const noexcept { return _last_command_count; }

    // The number of workflow commands the last `execute` visited per
    // `CommandKind` (indexed by `luisa::to_underlying(kind)`), which is what
    // the self test's per-kind summary reports.
    [[nodiscard]] luisa::span<const size_t> last_command_counts() const noexcept {
        return _command_counts;
    }

private:
    size_t _last_command_count{0u};
    // `vector(count, value)` in parentheses: the brace form would pick the
    // initializer-list constructor and make this a two-element vector.
    luisa::vector<size_t> _command_counts =
        luisa::vector<size_t>(command_kind_count, size_t{0u});
};

// ---------------------------------------------------------------------------
// sinks
// ---------------------------------------------------------------------------

// Writes `bytes` to `path` (creating parent directories). `overwrite == false`
// refuses to replace an existing file. `format == "png"` requires a full 2-D
// BYTE4/FLOAT4 image payload of `extent`.
[[nodiscard]] bool write_output(const luisa::filesystem::path &path,
                                luisa::span<const std::byte> bytes,
                                luisa::string_view format,
                                uint3 extent,
                                PixelStorage storage,
                                bool overwrite,
                                luisa::string &error) noexcept;

// Reads a whole file into `bytes` (bounded); false + message on failure.
[[nodiscard]] bool read_file(const luisa::filesystem::path &path,
                             size_t max_bytes,
                             luisa::vector<std::byte> &bytes,
                             luisa::string &error) noexcept;

}// namespace luisa::native_shader
