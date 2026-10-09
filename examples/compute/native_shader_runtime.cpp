// Runtime side of the native shader example: resource registry, shader
// registry, workflow executor and sinks.
//
// This translation unit owns every *device* interaction of the example: it
// creates the resources a dispatch document declares (loading their `input`
// through the DirectStorage extension when the backend provides one, otherwise
// from the host), translates the document's command list into `CommandList`s
// submitted to a `Stream`, downloads results into a stable-address arena and
// verifies them on the host. It never parses JSON and never decides what to run.
//
// Failure model: no `LUISA_ERROR`/`LUISA_ASSERT` and no exceptions. Every
// problem is reported through `Diagnostics` (or a returned `bool` plus an error
// string) so that one run can report as many problems as it can and the caller
// exits non-zero instead of the process aborting.
//
// Lifetime model (the two rules that keep this file from crashing):
//  * `BufferUploadCommand`/`TextureUploadCommand` keep a `const void *` and
//    `BufferDownloadCommand`/`TextureDownloadCommand` a `void *`, so every
//    payload must outlive the submit it was appended to. Upload payloads are
//    either the document's inline bytes (owned by `DispatchJson`, alive for the
//    whole run) or an arena block; download targets are always arena blocks,
//    and the arena lives until the end of `WorkflowExecutor::execute`.
//  * A `CommandList` must be committed before it is destroyed, so every flush
//    path commits the segment (an empty list commits harmlessly).
#include "native_shader_runtime.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include <luisa/backends/ext/dstorage_ext.hpp>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/rtx/aabb.h>
#include <luisa/runtime/rtx/triangle.h>

#include <stb/stb_image_write.h>

namespace luisa::native_shader {

using compute::AccelBuildCommand;
using compute::BindlessArrayUpdateCommand;
using compute::BufferCopyCommand;
using compute::BufferDownloadCommand;
using compute::BufferToTextureCopyCommand;
using compute::BufferUploadCommand;
using compute::ByteBuffer;
using compute::CommandList;
using compute::ComputeDispatchCmdEncoder;
using compute::CustomCommandUUID;
using compute::DStorageCompression;
using compute::DStorageExt;
using compute::DStorageFileView;
using compute::DStorageReadCommand;
using compute::MeshBuildCommand;
using compute::NativeShaderCompileInfo;
using compute::NativeShaderSourceType;
using compute::ProceduralPrimitiveBuildCommand;
using compute::TextureCopyCommand;
using compute::TextureDownloadCommand;
using compute::TextureToBufferCopyCommand;
using compute::TextureUploadCommand;
using compute::Volume;
using compute::VolumeView;

// ---------------------------------------------------------------------------
// element / storage / enum tables (the only place the JSON spellings live)
// ---------------------------------------------------------------------------

namespace {

// JSON enum spellings are matched after folding case and treating '-' as '_',
// so the document may write "float4", "Float-4" or "float_4" interchangeably.
// The folded name lives in a caller-provided stack buffer: none of the names
// below is longer than kEnumNameCapacity, and a longer name simply fails to
// match (it cannot be equal to any table entry).
constexpr auto kEnumNameCapacity = 64u;

[[nodiscard]] luisa::string_view fold_enum_name(luisa::string_view name,
                                                char *buffer) noexcept {
    auto count = std::min(name.size(), size_t{kEnumNameCapacity - 1u});
    for (auto i = 0u; i < count; i++) {
        auto c = name[i];
        if (c == '-') {
            c = '_';
        } else if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
        buffer[i] = c;
    }
    return luisa::string_view{buffer, count};
}

}// namespace

size_t buffer_element_size(BufferElement element) noexcept {
    switch (element) {
        case BufferElement::Float: return sizeof(float);
        case BufferElement::Float2: return sizeof(float2);
        case BufferElement::Float3: return sizeof(float3);
        case BufferElement::Float4: return sizeof(float4);
        case BufferElement::UInt: return sizeof(uint);
        case BufferElement::UInt2: return sizeof(uint2);
        case BufferElement::UInt3: return sizeof(uint3);
        case BufferElement::UInt4: return sizeof(uint4);
        case BufferElement::Int: return sizeof(int);
        case BufferElement::Int2: return sizeof(int2);
        case BufferElement::Int3: return sizeof(int3);
        case BufferElement::Int4: return sizeof(int4);
        case BufferElement::Byte: return sizeof(luisa::byte);
        case BufferElement::Triangle: return sizeof(compute::Triangle);
        case BufferElement::Aabb: return sizeof(compute::AABB);
    }
    return 0u;
}

luisa::string_view buffer_element_name(BufferElement element) noexcept {
    switch (element) {
        case BufferElement::Float: return "float";
        case BufferElement::Float2: return "float2";
        case BufferElement::Float3: return "float3";
        case BufferElement::Float4: return "float4";
        case BufferElement::UInt: return "uint";
        case BufferElement::UInt2: return "uint2";
        case BufferElement::UInt3: return "uint3";
        case BufferElement::UInt4: return "uint4";
        case BufferElement::Int: return "int";
        case BufferElement::Int2: return "int2";
        case BufferElement::Int3: return "int3";
        case BufferElement::Int4: return "int4";
        case BufferElement::Byte: return "byte";
        case BufferElement::Triangle: return "triangle";
        case BufferElement::Aabb: return "aabb";
    }
    return "unknown";
}

bool parse_buffer_element(luisa::string_view name, BufferElement &element) noexcept {
    struct Entry {
        luisa::string_view name;
        BufferElement value;
    };
    constexpr Entry table[] = {
        {"float", BufferElement::Float},
        {"float2", BufferElement::Float2},
        {"float3", BufferElement::Float3},
        {"float4", BufferElement::Float4},
        {"uint", BufferElement::UInt},
        {"uint2", BufferElement::UInt2},
        {"uint3", BufferElement::UInt3},
        {"uint4", BufferElement::UInt4},
        {"int", BufferElement::Int},
        {"int2", BufferElement::Int2},
        {"int3", BufferElement::Int3},
        {"int4", BufferElement::Int4},
        {"byte", BufferElement::Byte},
        {"triangle", BufferElement::Triangle},
        {"aabb", BufferElement::Aabb},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            element = entry.value;
            return true;
        }
    }
    return false;
}

luisa::string_view pixel_storage_name(PixelStorage storage) noexcept {
    switch (storage) {
        case PixelStorage::BYTE1: return "byte1";
        case PixelStorage::BYTE2: return "byte2";
        case PixelStorage::BYTE4: return "byte4";
        case PixelStorage::BYTE4_SRGB: return "byte4_srgb";
        case PixelStorage::SHORT1: return "short1";
        case PixelStorage::SHORT2: return "short2";
        case PixelStorage::SHORT4: return "short4";
        case PixelStorage::INT1: return "int1";
        case PixelStorage::INT2: return "int2";
        case PixelStorage::INT4: return "int4";
        case PixelStorage::HALF1: return "half1";
        case PixelStorage::HALF2: return "half2";
        case PixelStorage::HALF4: return "half4";
        case PixelStorage::FLOAT1: return "float1";
        case PixelStorage::FLOAT2: return "float2";
        case PixelStorage::FLOAT4: return "float4";
        case PixelStorage::R10G10B10A2: return "r10g10b10a2";
        case PixelStorage::R11G11B10: return "r11g11b10";
        default: break;// block-compressed storages have no document spelling
    }
    return "unknown";
}

bool parse_pixel_storage(luisa::string_view name, PixelStorage &storage) noexcept {
    struct Entry {
        luisa::string_view name;
        PixelStorage value;
    };
    // Only the uncompressed storages are expressible in the document: the
    // block-compressed ones are rejected by the semantic validator.
    constexpr Entry table[] = {
        {"byte1", PixelStorage::BYTE1},
        {"byte2", PixelStorage::BYTE2},
        {"byte4", PixelStorage::BYTE4},
        {"byte4_srgb", PixelStorage::BYTE4_SRGB},
        {"short1", PixelStorage::SHORT1},
        {"short2", PixelStorage::SHORT2},
        {"short4", PixelStorage::SHORT4},
        {"int1", PixelStorage::INT1},
        {"int2", PixelStorage::INT2},
        {"int4", PixelStorage::INT4},
        {"half1", PixelStorage::HALF1},
        {"half2", PixelStorage::HALF2},
        {"half4", PixelStorage::HALF4},
        {"float1", PixelStorage::FLOAT1},
        {"float2", PixelStorage::FLOAT2},
        {"float4", PixelStorage::FLOAT4},
        {"r10g10b10a2", PixelStorage::R10G10B10A2},
        {"r11g11b10", PixelStorage::R11G11B10},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            storage = entry.value;
            return true;
        }
    }
    return false;
}

luisa::string_view usage_name(Usage usage) noexcept {
    switch (usage) {
        case Usage::NONE: return "none";
        case Usage::READ: return "read";
        case Usage::WRITE: return "write";
        case Usage::READ_WRITE: return "read_write";
    }
    return "none";
}

bool parse_usage(luisa::string_view name, Usage &usage) noexcept {
    struct Entry {
        luisa::string_view name;
        Usage value;
    };
    constexpr Entry table[] = {
        {"none", Usage::NONE},
        {"read", Usage::READ},
        {"write", Usage::WRITE},
        {"read_write", Usage::READ_WRITE},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            usage = entry.value;
            return true;
        }
    }
    return false;
}

bool parse_slot_type(luisa::string_view name, BindlessSlotType &type) noexcept {
    struct Entry {
        luisa::string_view name;
        BindlessSlotType value;
    };
    // The document spells the slot type as in the wire format ("multiple",
    // "buffer", "texture2d", "texture3d"); the backend enumerator names are
    // accepted as aliases so that a document may also be written from the enum.
    constexpr Entry table[] = {
        {"multiple", BindlessSlotType::MULTIPLE},
        {"buffer", BindlessSlotType::BUFFER_ONLY},
        {"buffer_only", BindlessSlotType::BUFFER_ONLY},
        {"texture2d", BindlessSlotType::TEXTURE2D_ONLY},
        {"texture2d_only", BindlessSlotType::TEXTURE2D_ONLY},
        {"texture3d", BindlessSlotType::TEXTURE3D_ONLY},
        {"texture3d_only", BindlessSlotType::TEXTURE3D_ONLY},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            type = entry.value;
            return true;
        }
    }
    return false;
}

bool parse_accel_request(luisa::string_view name, AccelBuildRequest &request) noexcept {
    struct Entry {
        luisa::string_view name;
        AccelBuildRequest value;
    };
    constexpr Entry table[] = {
        {"prefer_update", AccelBuildRequest::PREFER_UPDATE},
        {"force_build", AccelBuildRequest::FORCE_BUILD},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            request = entry.value;
            return true;
        }
    }
    return false;
}

bool parse_sampler(const SamplerJson &json, Sampler &sampler) noexcept {
    struct Entry {
        luisa::string_view name;
        uint32_t value;
    };
    constexpr Entry filter_table[] = {
        {"point", luisa::to_underlying(Sampler::Filter::POINT)},
        {"linear_point", luisa::to_underlying(Sampler::Filter::LINEAR_POINT)},
        {"linear_linear", luisa::to_underlying(Sampler::Filter::LINEAR_LINEAR)},
        {"anisotropic", luisa::to_underlying(Sampler::Filter::ANISOTROPIC)},
    };
    constexpr Entry address_table[] = {
        {"edge", luisa::to_underlying(Sampler::Address::EDGE)},
        {"repeat", luisa::to_underlying(Sampler::Address::REPEAT)},
        {"mirror", luisa::to_underlying(Sampler::Address::MIRROR)},
        {"zero", luisa::to_underlying(Sampler::Address::ZERO)},
    };
    char buffer[2u][kEnumNameCapacity];
    auto filter = uint32_t{0u};
    auto address = uint32_t{0u};
    auto found_filter = false;
    auto found_address = false;
    auto filter_key = fold_enum_name(json.filter, buffer[0]);
    for (auto &&entry : filter_table) {
        if (entry.name == filter_key) {
            filter = entry.value;
            found_filter = true;
            break;
        }
    }
    auto address_key = fold_enum_name(json.address, buffer[1]);
    for (auto &&entry : address_table) {
        if (entry.name == address_key) {
            address = entry.value;
            found_address = true;
            break;
        }
    }
    if (!found_filter || !found_address) { return false; }
    sampler = Sampler{static_cast<Sampler::Filter>(filter),
                      static_cast<Sampler::Address>(address)};
    return true;
}

luisa::string_view native_shader_language_name(NativeShaderLanguage language) noexcept {
    switch (language) {
        case NativeShaderLanguage::HLSL: return "hlsl";
        case NativeShaderLanguage::GLSL: return "glsl";
        case NativeShaderLanguage::CUDA_NVRTC: return "cuda_nvrtc";
    }
    return "unknown";
}

bool parse_native_shader_language(luisa::string_view name,
                                  NativeShaderLanguage &language) noexcept {
    struct Entry {
        luisa::string_view name;
        NativeShaderLanguage value;
    };
    constexpr Entry table[] = {
        {"hlsl", NativeShaderLanguage::HLSL},
        {"glsl", NativeShaderLanguage::GLSL},
        {"cuda_nvrtc", NativeShaderLanguage::CUDA_NVRTC},
        {"cuda", NativeShaderLanguage::CUDA_NVRTC},
        {"cuda_cxx", NativeShaderLanguage::CUDA_NVRTC},
    };
    char buffer[kEnumNameCapacity];
    auto key = fold_enum_name(name, buffer);
    for (auto &&entry : table) {
        if (entry.name == key) {
            language = entry.value;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

void Diagnostics::error(luisa::string message) {
    // Errors are the reason the example exits non-zero, so they are bounded but
    // never dropped silently before the bound is reached.
    if (errors.size() >= max_errors) { return; }
    errors.emplace_back(std::move(message));
}

void Diagnostics::warning(luisa::string message) {
    // Warnings are uncapped, but a repeated consecutive message adds nothing.
    if (!warnings.empty() && warnings.back() == message) { return; }
    warnings.emplace_back(std::move(message));
}

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

namespace {

// Every `filesystem::path` in this file is built through `path_from_narrow`
// (or `to_string`), never through the narrow `path` constructor, which decodes
// the bytes with the ANSI code page and terminates the process on input that
// page cannot represent.
[[nodiscard]] bool assign_path(luisa::string_view text,
                               luisa::filesystem::path &path) noexcept {
    path.clear();
    return luisa::path_from_narrow(text, path);
}

}// namespace

void PathResolver::set_document_dir(luisa::string_view directory) noexcept {
    if (!assign_path(directory, _document_dir) && _diagnostics != nullptr) {
        _diagnostics->error(luisa::format(
            "the document directory '{}' is not a decodable path", directory));
    }
}

void PathResolver::set_workdir(luisa::string_view directory) noexcept {
    if (!assign_path(directory, _workdir)) {
        if (_diagnostics != nullptr && !directory.empty()) {
            _diagnostics->error(luisa::format(
                "the working directory '{}' is not a decodable path", directory));
        }
        _has_workdir = false;
        return;
    }
    _has_workdir = !_workdir.empty();
}

void PathResolver::set_output_dir(luisa::string_view directory) noexcept {
    if (!assign_path(directory, _output_dir) && _diagnostics != nullptr && !directory.empty()) {
        _diagnostics->error(luisa::format(
            "the output directory '{}' is not a decodable path", directory));
    }
}

luisa::filesystem::path PathResolver::resolve_input(luisa::string_view path) const noexcept {
    luisa::filesystem::path resolved;
    if (!assign_path(path, resolved)) { return {}; }
    if (resolved.is_absolute()) { return resolved; }
    // Input paths are relative to the working directory when one was given,
    // otherwise to the directory of the document itself.
    auto &&base = _has_workdir ? _workdir : _document_dir;
    if (base.empty()) { return resolved; }
    return base / resolved;
}

luisa::filesystem::path PathResolver::resolve_output(luisa::string_view path) const noexcept {
    luisa::filesystem::path resolved;
    if (!assign_path(path, resolved)) { return {}; }
    if (resolved.is_absolute() || _output_dir.empty()) { return resolved; }
    return _output_dir / resolved;
}

luisa::filesystem::path PathResolver::resolve_shader(luisa::string_view path) const noexcept {
    // A shader source is an input of the document, so it follows the same rule.
    return resolve_input(path);
}

bool PathResolver::ensure_parent_directory(const luisa::filesystem::path &path,
                                           luisa::string &error) noexcept {
    auto parent = path.parent_path();
    // A bare file name has no parent: the current working directory is used and
    // it necessarily exists.
    if (parent.empty()) { return true; }
    std::error_code ec;
    luisa::filesystem::create_directories(parent, ec);
    if (ec) {
        error = luisa::format("cannot create the directory '{}': {}",
                              luisa::to_string(parent), ec.message());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// file-local helpers
// ---------------------------------------------------------------------------

namespace {

// ---- names ----------------------------------------------------------------

// Compares a document spelling with a canonical (already folded) name.
[[nodiscard]] bool enum_name_is(luisa::string_view name,
                                luisa::string_view expected) noexcept {
    char buffer[kEnumNameCapacity];
    return fold_enum_name(name, buffer) == expected;
}

// ---- stable-address payloads ----------------------------------------------

// Hands out byte blocks whose address never changes, which is what the upload
// and download commands need (they store raw pointers). One `execute` owns one
// arena and keeps it alive until the last synchronize of the frame.
class ByteArena {

private:
    luisa::vector<luisa::unique_ptr<std::byte[]>> _blocks;

public:
    [[nodiscard]] std::byte *allocate(size_t byte_size) noexcept {
        if (byte_size == 0u) { return nullptr; }
        // `eastl::make_unique<std::byte[]>` writes the hidden array header
        // that `eastl::default_delete<std::byte[]>` reads back, so the block
        // must be allocated with it (a bare `new std::byte[]` would be freed
        // as if it had that header).
        auto block = luisa::make_unique<std::byte[]>(byte_size);
        auto *pointer = block.get();
        _blocks.emplace_back(std::move(block));
        return pointer;
    }
};

// ---- scalar / element type dispatch ---------------------------------------

// The three element types `create_image<T>` / `create_volume<T>` accept.
enum class ScalarKind : uint32_t {
    Float,
    UInt,
    Int,
};

template<typename F>
[[nodiscard]] bool each_scalar_type(ScalarKind kind, F &&f) noexcept {
    switch (kind) {
        case ScalarKind::Float: return f.template operator()<float>();
        case ScalarKind::UInt: return f.template operator()<uint>();
        case ScalarKind::Int: return f.template operator()<int>();
    }
    return false;
}

// One instantiation per `BufferElement`; `f.template operator()<T>()` creates
// the typed resource (or typed view) for that element.
template<typename F>
[[nodiscard]] bool each_buffer_element(BufferElement element, F &&f) noexcept {
    switch (element) {
        case BufferElement::Float: return f.template operator()<float>();
        case BufferElement::Float2: return f.template operator()<float2>();
        case BufferElement::Float3: return f.template operator()<float3>();
        case BufferElement::Float4: return f.template operator()<float4>();
        case BufferElement::UInt: return f.template operator()<uint>();
        case BufferElement::UInt2: return f.template operator()<uint2>();
        case BufferElement::UInt3: return f.template operator()<uint3>();
        case BufferElement::UInt4: return f.template operator()<uint4>();
        case BufferElement::Int: return f.template operator()<int>();
        case BufferElement::Int2: return f.template operator()<int2>();
        case BufferElement::Int3: return f.template operator()<int3>();
        case BufferElement::Int4: return f.template operator()<int4>();
        case BufferElement::Byte: return f.template operator()<luisa::byte>();
        case BufferElement::Triangle: return f.template operator()<compute::Triangle>();
        case BufferElement::Aabb: return f.template operator()<compute::AABB>();
    }
    return false;
}

// The scalar channel type an image/volume resource was created with: the
// declared `element` wins, otherwise the storage (float* -> float, int* -> int,
// everything else -> uint).
[[nodiscard]] ScalarKind scalar_kind_of(const ResourceJson &spec) noexcept {
    BufferElement element{};
    if (!spec.element.empty() && parse_buffer_element(spec.element, element)) {
        switch (element) {
            case BufferElement::Float:
            case BufferElement::Float2:
            case BufferElement::Float3:
            case BufferElement::Float4:
                return ScalarKind::Float;
            case BufferElement::Int:
            case BufferElement::Int2:
            case BufferElement::Int3:
            case BufferElement::Int4:
                return ScalarKind::Int;
            default: return ScalarKind::UInt;
        }
    }
    PixelStorage storage{};
    if (!spec.storage.empty() && parse_pixel_storage(spec.storage, storage)) {
        switch (storage) {
            case PixelStorage::FLOAT1:
            case PixelStorage::FLOAT2:
            case PixelStorage::FLOAT4:
                return ScalarKind::Float;
            case PixelStorage::INT1:
            case PixelStorage::INT2:
            case PixelStorage::INT4:
                return ScalarKind::Int;
            default: return ScalarKind::UInt;
        }
    }
    return ScalarKind::UInt;
}

// ---- file regions ---------------------------------------------------------

// Resolves the region `[offset, offset + size_in)` of `path`, where a requested
// size of 0 means "the rest of the file". `max_size` is the byte size of the
// destination the region is loaded into and bounds the result. Every message
// states the numbers it compared so that a failure can be diagnosed from the
// log alone.
[[nodiscard]] bool resolve_file_region(const luisa::filesystem::path &path,
                                       size_t offset, size_t requested_size,
                                       size_t max_size, size_t &file_offset,
                                       size_t &size, luisa::string &error) noexcept {
    namespace fs = luisa::filesystem;
    std::error_code ec;
    auto exists = fs::exists(path, ec);
    if (ec) {
        error = luisa::format("cannot read the status of the input file '{}': {}",
                              luisa::to_string(path), ec.message());
        return false;
    }
    if (!exists) {
        error = luisa::format("the input file '{}' does not exist",
                              luisa::to_string(path));
        return false;
    }
    if (!fs::is_regular_file(path, ec) || ec) {
        error = luisa::format("the input path '{}' is not a regular file",
                              luisa::to_string(path));
        return false;
    }
    auto file_size = fs::file_size(path, ec);
    if (ec) {
        error = luisa::format("cannot determine the size of the input file '{}'",
                              luisa::to_string(path));
        return false;
    }
    if (file_size == 0u) {
        error = luisa::format("the input file '{}' is empty",
                              luisa::to_string(path));
        return false;
    }
    if (offset >= file_size) {
        error = luisa::format("the input offset {} is not below the size {} of '{}'",
                              offset, file_size, luisa::to_string(path));
        return false;
    }
    auto available = static_cast<size_t>(file_size) - offset;
    auto size_in = requested_size == 0u ? available : requested_size;
    if (size_in > available) {
        error = luisa::format("the input region [{}, {}) exceeds the {} bytes of "
                              "'{}' that remain after its offset",
                              offset, offset + size_in, available,
                              luisa::to_string(path));
        return false;
    }
    if (size_in > max_size) {
        error = luisa::format("the input region [{}, {}) of '{}' is {} bytes, "
                              "which exceeds the {} bytes of the destination",
                              offset, offset + size_in, luisa::to_string(path),
                              size_in, max_size);
        return false;
    }
    file_offset = offset;
    size = size_in;
    return true;
}

// Reads exactly `size` bytes at `offset` of `path` into `destination`.
[[nodiscard]] bool read_file_region(const luisa::filesystem::path &path, size_t offset,
                                    size_t size, std::byte *destination,
                                    luisa::string &error) noexcept {
    if (size == 0u) { return true; }
    if (destination == nullptr) {
        error = luisa::format("internal error: no staging buffer for the {} bytes of '{}'",
                              size, luisa::to_string(path));
        return false;
    }
    std::ifstream file{path, std::ios::in | std::ios::binary};
    if (!file.is_open()) {
        error = luisa::format("cannot open the input file '{}'",
                              luisa::to_string(path));
        return false;
    }
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file.read(reinterpret_cast<char *>(destination),
                   static_cast<std::streamsize>(size))) {
        error = luisa::format("cannot read {} bytes at offset {} of '{}'",
                              size, offset, luisa::to_string(path));
        return false;
    }
    return true;
}

// ---- byte regions --------------------------------------------------------

// Resolves a byte region of a resource or a file: a requested size of 0 means
// "the rest from `offset`". `total` is the byte size of the source, `limit` the
// byte size of whatever the region is copied into.
[[nodiscard]] bool resolve_resource_region(size_t offset, size_t requested_size,
                                           size_t total, size_t limit,
                                           size_t &resolved_offset, size_t &resolved_size,
                                           luisa::string_view what,
                                           luisa::string &error) noexcept {
    if (offset > total) {
        error = luisa::format("the offset {} of {} exceeds its {} bytes",
                              offset, what, total);
        return false;
    }
    auto available = total - offset;
    auto size = requested_size == 0u ? available : requested_size;
    if (size > available) {
        error = luisa::format("the region [{}, {}) of {} exceeds its {} remaining bytes",
                              offset, offset + size, what, available);
        return false;
    }
    if (size > limit) {
        error = luisa::format("the region [{}, {}) of {} is {} bytes, which exceeds the {} bytes "
                              "of the destination",
                              offset, offset + size, what, size, limit);
        return false;
    }
    resolved_offset = offset;
    resolved_size = size;
    return true;
}

// ---- direct storage -------------------------------------------------------

// How the file inputs of one run reached the device. The extension performs the
// *first* read of a file region; a repeat request for the same region (the same
// file, offset and size, already handed to DirectStorage once) is served from
// the host instead. The first host read emits exactly one warning (an error
// under `--strict`) that names the backend, so a backend without the extension
// is reported once instead of once per resource.
struct DStorageUseState {
    bool warned{false};
    luisa::vector<luisa::string> regions;
};

[[nodiscard]] DStorageUseState &dstorage_use_state() noexcept {
    static DStorageUseState state;
    return state;
}

// Claims `path`'s region for the DirectStorage path; false for a repeat.
[[nodiscard]] bool claim_dstorage_region(const luisa::filesystem::path &path,
                                         size_t offset, size_t size) noexcept {
    auto key = luisa::format("{}:{}+{}", luisa::to_string(path), offset, size);
    auto &&state = dstorage_use_state();
    for (auto &&region : state.regions) {
        if (region == key) { return false; }
    }
    state.regions.emplace_back(std::move(key));
    return true;
}

// Reports the single "this load went through the host" event.
void report_host_fallback(luisa::string_view reason, luisa::string_view backend,
                          bool strict, Diagnostics &diagnostics) noexcept {
    auto &&state = dstorage_use_state();
    if (state.warned) { return; }
    state.warned = true;
    if (strict) {
        diagnostics.error(luisa::format(
            "loading input files on the host ({}): backend '{}' cannot use the "
            "DirectStorage extension for this region",
            reason, backend));
    } else {
        LUISA_WARNING("loading input files on the host ({}): backend '{}' cannot "
                      "use the DirectStorage extension for this region",
                      reason, backend);
    }
}

// Decides how one file region reaches the device: true when it is read through
// the DirectStorage extension, false when the host path must be used (the
// one-time host fallback event is reported in that case, `reason` saying why).
[[nodiscard]] bool choose_dstorage(bool enabled, DStorageExt *ext, Stream *stream,
                                   bool compression_supported,
                                   luisa::string_view compression_name,
                                   const luisa::filesystem::path &path,
                                   size_t offset, size_t size,
                                   luisa::string_view backend, bool strict,
                                   Diagnostics &diagnostics) noexcept {
    if (!enabled) {
        report_host_fallback("'dstorage.enabled' is false", backend, strict, diagnostics);
        return false;
    }
    if (ext == nullptr) {
        report_host_fallback("the backend has no DirectStorage extension", backend, strict, diagnostics);
        return false;
    }
    if (stream == nullptr) {
        report_host_fallback("no DirectStorage stream is available", backend, strict, diagnostics);
        return false;
    }
    if (!compression_supported) {
        report_host_fallback(luisa::format("the compression '{}' is not supported",
                                           compression_name),
                             backend, strict, diagnostics);
        return false;
    }
    if (!claim_dstorage_region(path, offset, size)) {
        report_host_fallback("this file region was already read through DirectStorage",
                             backend, strict, diagnostics);
        return false;
    }
    return true;
}

// Parses a document compression spelling ("none" | "gdeflate"); the input's own
// spelling wins over `config.dstorage.compression`, which is the default for
// the whole document.
[[nodiscard]] bool parse_dstorage_compression(luisa::string_view name,
                                              DStorageCompression &compression) noexcept {
    if (enum_name_is(name, "none")) {
        compression = DStorageCompression::None;
        return true;
    }
    if (enum_name_is(name, "gdeflate")) {
        compression = DStorageCompression::GDeflate;
        return true;
    }
    return false;
}

// The `DStorageReadCommand` that fills `entry`'s resource from `view`. The
// destination is reconstructed from the flat metadata because the runtime
// wrappers are type-erased: a buffer becomes a `BufferView<std::byte>` over the
// destination region, an image or volume a typed view of the level and extent
// the read targets. Returns null when the resource kind cannot be a
// destination.
[[nodiscard]] luisa::unique_ptr<compute::DStorageReadCommand>
build_dstorage_read(DStorageFileView view, const ResourceRegistry::Entry &entry,
                    size_t destination_offset, uint32_t level, uint3 texture_size,
                    DStorageCompression compression) noexcept {
    auto &&resource = entry.resource;
    switch (entry.spec.type) {
        case ResourceType::Buffer: {
            BufferView<std::byte> destination{
                resource.native_handle, resource.handle, 1u, destination_offset,
                view.size_bytes(), resource.byte_size};
            return view.copy_to(destination, compression);
        }
        case ResourceType::Texture: {
            luisa::unique_ptr<compute::DStorageReadCommand> command;
            auto built = each_scalar_type(scalar_kind_of(entry.spec), [&]<typename T>() noexcept {
                compute::ImageView<T> destination{resource.native_handle, resource.handle,
                                                  resource.storage, level,
                                                  uint2{texture_size.x, texture_size.y}};
                command = view.copy_to(destination, compression);
                return true;
            });
            return built ? std::move(command) : nullptr;
        }
        case ResourceType::Volume: {
            luisa::unique_ptr<compute::DStorageReadCommand> command;
            auto built = each_scalar_type(scalar_kind_of(entry.spec), [&]<typename T>() noexcept {
                compute::VolumeView<T> destination{resource.native_handle, resource.handle,
                                                   resource.storage, level, texture_size};
                command = view.copy_to(destination, compression);
                return true;
            });
            return built ? std::move(command) : nullptr;
        }
        default: return nullptr;
    }
}

// ---- misc -----------------------------------------------------------------

// Halves `value` `times` times, saturating at one (a shifted extent never
// becomes zero: 32-bit shifts of the level are not representable otherwise).
[[nodiscard]] uint32_t mip_dimension(uint32_t value, uint32_t times) noexcept {
    auto result = std::max(1u, value);
    for (auto i = 0u; i < times && result > 1u; i++) { result /= 2u; }
    return result;
}

[[nodiscard]] uint3 mip_extent(uint3 extent, uint32_t level) noexcept {
    return uint3{mip_dimension(extent.x, level),
                 mip_dimension(extent.y, level),
                 mip_dimension(extent.z, level)};
}

}// namespace

// ---------------------------------------------------------------------------
// sinks
// ---------------------------------------------------------------------------

bool write_output(const luisa::filesystem::path &path,
                  luisa::span<const std::byte> bytes,
                  luisa::string_view format,
                  uint3 extent,
                  PixelStorage storage,
                  bool overwrite,
                  luisa::string &error) noexcept {
    namespace fs = luisa::filesystem;
    std::error_code ec;
    if (!overwrite && fs::exists(path, ec) && !ec) {
        error = luisa::format("refusing to overwrite the existing file '{}'",
                              luisa::to_string(path));
        return false;
    }
    auto parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) {
            error = luisa::format("cannot create the directory '{}': {}",
                                  luisa::to_string(parent), ec.message());
            return false;
        }
    }
    if (enum_name_is(format, "png")) {
        if (extent.z > 1u) {
            error = luisa::format("a PNG sink only accepts a 2-D image, got a "
                                  "{}x{}x{} payload",
                                  extent.x, extent.y, extent.z);
            return false;
        }
        if (storage != PixelStorage::BYTE4 && storage != PixelStorage::BYTE4_SRGB &&
            storage != PixelStorage::FLOAT4) {
            error = luisa::format("a PNG sink needs byte4, byte4_srgb or float4 "
                                  "pixels, got '{}'",
                                  pixel_storage_name(storage));
            return false;
        }
        auto expected = compute::pixel_storage_size(storage, extent);
        if (bytes.size() != expected) {
            error = luisa::format("a PNG sink needs the whole {}x{} '{}' image "
                                  "({} bytes), got {} bytes",
                                  extent.x, extent.y, pixel_storage_name(storage),
                                  expected, bytes.size());
            return false;
        }
        auto pixel_count = static_cast<size_t>(extent.x) * extent.y;
        const void *payload = bytes.data();
        luisa::vector<std::byte> converted;
        if (storage == PixelStorage::FLOAT4) {
            // Linear HDR values scaled by 255 and clamped: the sink writes the
            // bytes as they are, it does not tone map.
            converted.resize(pixel_count * 4u);
            auto *source = reinterpret_cast<const float *>(bytes.data());
            for (auto i = size_t{0u}; i < converted.size(); i++) {
                auto value = std::clamp(source[i], 0.0f, 1.0f) * 255.0f + 0.5f;
                converted[i] = static_cast<std::byte>(static_cast<uint8_t>(value));
            }
            payload = converted.data();
        }
        auto name = luisa::to_string(path);
        if (stbi_write_png(name.c_str(), static_cast<int>(extent.x),
                           static_cast<int>(extent.y), 4, payload, 0) == 0) {
            error = luisa::format("cannot write the PNG file '{}'",
                                  luisa::to_string(path));
            return false;
        }
        return true;
    }
    std::ofstream file{path, std::ios::out | std::ios::binary | std::ios::trunc};
    if (!file.is_open()) {
        error = luisa::format("cannot open '{}' for writing",
                              luisa::to_string(path));
        return false;
    }
    if (!bytes.empty()) {
        file.write(reinterpret_cast<const char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    if (!file) {
        error = luisa::format("cannot write {} bytes to '{}'",
                              bytes.size(), luisa::to_string(path));
        return false;
    }
    return true;
}

bool read_file(const luisa::filesystem::path &path, size_t max_bytes,
               luisa::vector<std::byte> &bytes, luisa::string &error) noexcept {
    bytes.clear();
    size_t offset = 0u;
    size_t size = 0u;
    if (!resolve_file_region(path, 0u, 0u, max_bytes, offset, size, error)) {
        return false;
    }
    bytes.resize(size);
    if (!read_file_region(path, offset, size, bytes.data(), error)) {
        bytes.clear();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

namespace {

constexpr auto kResourcePending = uint8_t{0u};
constexpr auto kResourceCreated = uint8_t{1u};
constexpr auto kResourceFailed = uint8_t{2u};
constexpr auto kResourceLoaded = uint8_t{3u};

// The compression a file input is read with: the input's own spelling wins over
// `config.dstorage.compression`, which is the document-wide default (the wire
// format cannot distinguish an absent key from the explicit "none").
[[nodiscard]] luisa::string_view effective_compression(const InputJson &input,
                                                       const ConfigJson &config) noexcept {
    if (!input.compression.empty() && !enum_name_is(input.compression, "none")) {
        return input.compression;
    }
    if (!config.dstorage.compression.empty() &&
        !enum_name_is(config.dstorage.compression, "none")) {
        return config.dstorage.compression;
    }
    return "none";
}

}// namespace

const ResourceRegistry::Entry *ResourceRegistry::find(luisa::string_view name) const noexcept {
    auto it = _index.find(luisa::string{name});
    return it == _index.end() ? nullptr : &_entries[it->second];
}

bool ResourceRegistry::create_all(Device &device, Stream &stream, Stream *dstorage_stream,
                                  const DispatchJson &document, const PathResolver &paths,
                                  Diagnostics &diagnostics) noexcept {
    _entries.clear();
    _index.clear();
    _entries.reserve(document.resources.size());
    // One entry per document resource, so that every message can name the
    // document index. Duplicate names are the semantic validator's business; a
    // lookup always resolves to the first definition.
    for (auto i = 0u; i < document.resources.size(); i++) {
        auto &&spec = document.resources[i];
        _index.try_emplace(spec.name, static_cast<size_t>(i));
        _entries.push_back(Entry{spec, OwnedResource{}});
    }
    auto count = _entries.size();
    if (count == 0u) { return true; }

    auto error_at = [&](size_t index, luisa::string message) noexcept {
        diagnostics.error(luisa::format("resources[{}] ({}): {}", index,
                                        _entries[index].spec.name, message));
    };
    auto index_of = [&](luisa::string_view name) noexcept -> size_t {
        auto it = _index.find(luisa::string{name});
        return it == _index.end() ? count : it->second;
    };

    // ---- pass 1: create in dependency order -------------------------------
    enum class Outcome : uint32_t {
        Created,
        Deferred,
        Failed,
    };
    luisa::vector<uint8_t> state(count, kResourcePending);
    auto *dstorage = device.extension<DStorageExt>();
    auto try_create = [&](size_t index) noexcept -> Outcome {
        auto &&entry = _entries[index];
        auto &&spec = entry.spec;
        auto &&resource = entry.resource;
        switch (spec.type) {
            case ResourceType::Buffer: {
                if (spec.element.empty()) {
                    if (spec.byte_size == 0u) {
                        error_at(index, "a byte buffer needs a nonzero byte_size");
                        return Outcome::Failed;
                    }
                    auto buffer = device.create_byte_buffer(spec.byte_size);
                    auto view = buffer.view();
                    resource.handle = view.handle();
                    resource.native_handle = view.native_handle();
                    resource.stride = 1u;
                    resource.byte_size = view.total_size_bytes();
                    resource.owner = Owner::create(std::move(buffer));
                    return Outcome::Created;
                }
                BufferElement element{};
                if (!parse_buffer_element(spec.element, element)) {
                    error_at(index, luisa::format("unknown buffer element '{}'", spec.element));
                    return Outcome::Failed;
                }
                if (spec.count == 0u) {
                    error_at(index, "a typed buffer needs a nonzero count");
                    return Outcome::Failed;
                }
                auto created = each_buffer_element(element, [&]<typename T>() noexcept {
                    auto buffer = device.create_buffer<T>(spec.count);
                    auto view = buffer.view();
                    resource.handle = view.handle();
                    resource.native_handle = view.native_handle();
                    resource.stride = view.stride();
                    resource.byte_size = view.total_size_bytes();
                    resource.owner = Owner::create(std::move(buffer));
                    return true;
                });
                if (!created) {
                    error_at(index, "internal error: unhandled buffer element");
                    return Outcome::Failed;
                }
                return Outcome::Created;
            }
            case ResourceType::Texture:
            case ResourceType::Volume: {
                PixelStorage storage{};
                if (!parse_pixel_storage(spec.storage, storage)) {
                    error_at(index, luisa::format("unknown pixel storage '{}'", spec.storage));
                    return Outcome::Failed;
                }
                if (compute::is_block_compressed(storage)) {
                    error_at(index, luisa::format("block-compressed storage '{}' is not supported",
                                                  spec.storage));
                    return Outcome::Failed;
                }
                auto is_volume = spec.type == ResourceType::Volume;
                if (spec.size.x == 0u || spec.size.y == 0u || (is_volume && spec.size.z == 0u)) {
                    error_at(index, "the extent must be nonzero in every dimension");
                    return Outcome::Failed;
                }
                auto extent = is_volume ? spec.size : uint3{spec.size.x, spec.size.y, 1u};
                auto levels = std::max(1u, spec.levels);
                resource.storage = storage;
                resource.extent = extent;
                resource.levels = levels;
                resource.byte_size = compute::pixel_storage_size(storage, extent);
                auto kind = scalar_kind_of(spec);
                if (is_volume) {
                    auto created = each_scalar_type(kind, [&]<typename T>() noexcept {
                        auto volume = device.create_volume<T>(storage, extent, levels);
                        auto view = volume.view(0u);
                        resource.handle = view.handle();
                        resource.native_handle = view.native_handle();
                        resource.owner = Owner::create(std::move(volume));
                        return true;
                    });
                    if (!created) {
                        error_at(index, "internal error: unhandled volume element type");
                        return Outcome::Failed;
                    }
                } else {
                    auto created = each_scalar_type(kind, [&]<typename T>() noexcept {
                        auto image = device.create_image<T>(storage, uint2{extent.x, extent.y}, levels);
                        auto view = image.view(0u);
                        resource.handle = view.handle();
                        resource.native_handle = view.native_handle();
                        if constexpr (std::is_same_v<T, float>) {
                            // The ImGui window binds an `Image<float>` by
                            // reference, so the registry publishes the address
                            // of the heap object its owner keeps alive: the
                            // holder is moved into the `Owner`, the image it
                            // points at never moves.
                            auto holder = luisa::make_unique<Image<float>>(std::move(image));
                            resource.float_image = holder.get();
                            resource.owner = Owner::create(std::move(holder));
                        } else {
                            resource.owner = Owner::create(std::move(image));
                        }
                        return true;
                    });
                    if (!created) {
                        error_at(index, "internal error: unhandled image element type");
                        return Outcome::Failed;
                    }
                }
                return Outcome::Created;
            }
            case ResourceType::BindlessArray: {
                if (spec.slot_count == 0u) {
                    error_at(index, "a bindless array needs a nonzero slot_count");
                    return Outcome::Failed;
                }
                BindlessSlotType type{};
                if (!parse_slot_type(spec.slot_type, type)) {
                    error_at(index, luisa::format("unknown bindless slot type '{}'", spec.slot_type));
                    return Outcome::Failed;
                }
                auto array = device.create_bindless_array(spec.slot_count, type);
                resource.handle = array.handle();
                resource.native_handle = array.native_handle();
                resource.owner = Owner::create(std::move(array));
                return Outcome::Created;
            }
            case ResourceType::Accel: {
                auto accel = device.create_accel();
                resource.handle = accel.handle();
                resource.native_handle = accel.native_handle();
                resource.owner = Owner::create(std::move(accel));
                return Outcome::Created;
            }
            case ResourceType::Mesh: {
                auto vertex_index = index_of(spec.vertex_buffer);
                auto triangle_index = index_of(spec.triangle_buffer);
                if (vertex_index == count || triangle_index == count) {
                    error_at(index, luisa::format(
                                        "the mesh needs the resources '{}' (vertices) and '{}' "
                                        "(triangles), and the document does not define both",
                                        spec.vertex_buffer, spec.triangle_buffer));
                    return Outcome::Failed;
                }
                if (state[vertex_index] != kResourceCreated ||
                    state[triangle_index] != kResourceCreated) {
                    return Outcome::Deferred;
                }
                auto &&vertices = _entries[vertex_index];
                auto &&triangles = _entries[triangle_index];
                if (vertices.spec.element.empty()) {
                    error_at(index, luisa::format("the vertex buffer '{}' must be a typed buffer",
                                                  spec.vertex_buffer));
                    return Outcome::Failed;
                }
                if (triangles.spec.element.empty()) {
                    error_at(index, luisa::format("the triangle buffer '{}' must be a typed buffer",
                                                  spec.triangle_buffer));
                    return Outcome::Failed;
                }
                BufferElement vertex_element{};
                if (!parse_buffer_element(vertices.spec.element, vertex_element)) {
                    error_at(index, luisa::format("the vertex buffer '{}' has the unknown element '{}'",
                                                  spec.vertex_buffer, vertices.spec.element));
                    return Outcome::Failed;
                }
                auto vertex_stride = vertices.resource.stride;
                auto vertex_bytes = vertices.resource.byte_size;
                auto triangle_bytes = triangles.resource.byte_size;
                if (triangle_bytes < sizeof(compute::Triangle)) {
                    error_at(index, luisa::format("the triangle buffer '{}' holds {} bytes, which is "
                                                  "less than one triangle",
                                                  spec.triangle_buffer, triangle_bytes));
                    return Outcome::Failed;
                }
                BufferView<compute::Triangle> triangle_view{
                    triangles.resource.native_handle, triangles.resource.handle,
                    sizeof(compute::Triangle), 0u,
                    triangle_bytes / sizeof(compute::Triangle),
                    triangle_bytes / sizeof(compute::Triangle)};
                auto created = each_buffer_element(vertex_element, [&]<typename V>() noexcept {
                    BufferView<V> vertex_view{vertices.resource.native_handle,
                                              vertices.resource.handle, sizeof(V), 0u,
                                              vertex_bytes / sizeof(V), vertex_bytes / sizeof(V)};
                    auto mesh = device.create_mesh(vertex_view, vertex_stride, triangle_view);
                    resource.handle = mesh.handle();
                    resource.native_handle = mesh.native_handle();
                    resource.owner = Owner::create(std::move(mesh));
                    return true;
                });
                if (!created) {
                    error_at(index, "internal error: unhandled vertex element type");
                    return Outcome::Failed;
                }
                return Outcome::Created;
            }
            case ResourceType::ProceduralPrimitive: {
                // `aabb_buffer` is the creation-time AABB range of the
                // primitive; the `procedural_primitive_build` command carries
                // the buffer its BLAS is built from explicitly.
                BufferView<compute::AABB> aabb_view;
                if (!spec.aabb_buffer.empty()) {
                    auto boxes_index = index_of(spec.aabb_buffer);
                    if (boxes_index == count) {
                        error_at(index, luisa::format(
                                            "aabb_buffer: the resource '{}' is not defined",
                                            spec.aabb_buffer));
                        return Outcome::Failed;
                    }
                    if (state[boxes_index] != kResourceCreated) { return Outcome::Deferred; }
                    auto &&boxes = _entries[boxes_index];
                    BufferElement element{};
                    if (boxes.spec.element.empty() ||
                        !parse_buffer_element(boxes.spec.element, element) ||
                        element != BufferElement::Aabb) {
                        error_at(index, luisa::format(
                                            "aabb_buffer: '{}' must have the element 'aabb'",
                                            spec.aabb_buffer));
                        return Outcome::Failed;
                    }
                    auto box_bytes = boxes.resource.byte_size;
                    aabb_view = BufferView<compute::AABB>{boxes.resource.native_handle,
                                                          boxes.resource.handle,
                                                          sizeof(compute::AABB), 0u,
                                                          box_bytes / sizeof(compute::AABB),
                                                          box_bytes / sizeof(compute::AABB)};
                }
                auto primitive = device.create_procedural_primitive(aabb_view);
                resource.handle = primitive.handle();
                resource.native_handle = primitive.native_handle();
                resource.owner = Owner::create(std::move(primitive));
                return Outcome::Created;
            }
        }
        error_at(index, "internal error: unhandled resource type");
        return Outcome::Failed;
    };

    auto ok = true;
    auto pending = count;
    while (pending > 0u) {
        auto progress = false;
        for (auto i = size_t{0u}; i < count; i++) {
            if (state[i] != kResourcePending) { continue; }
            switch (try_create(i)) {
                case Outcome::Deferred: continue;
                case Outcome::Created:
                    state[i] = kResourceCreated;
                    break;
                case Outcome::Failed:
                    state[i] = kResourceFailed;
                    ok = false;
                    break;
            }
            pending--;
            progress = true;
        }
        if (!progress) {
            // Nothing could be created in a whole round: the remaining
            // resources wait for each other.
            luisa::string names;
            for (auto i = size_t{0u}; i < count; i++) {
                if (state[i] != kResourcePending) { continue; }
                if (!names.empty()) { names.append(", "); }
                names.append(luisa::format("'{}'", _entries[i].spec.name));
            }
            diagnostics.error(luisa::format("resources: dependency cycle among {}", names));
            ok = false;
            break;
        }
    }

    // ---- pass 2: load the `input` of every created resource ---------------
    ByteArena staging;
    for (auto i = size_t{0u}; i < count; i++) {
        if (state[i] != kResourceCreated) { continue; }
        auto &&entry = _entries[i];
        auto &&input = entry.spec.input;
        if (input.kind == InputJson::Kind::None) {
            state[i] = kResourceLoaded;
            continue;
        }
        auto &&resource = entry.resource;
        auto byte_addressable = entry.spec.type == ResourceType::Buffer;
        auto texture_like = entry.spec.type == ResourceType::Texture ||
                            entry.spec.type == ResourceType::Volume;
        switch (input.kind) {
            case InputJson::Kind::None: break;
            case InputJson::Kind::Inline: {
                auto &&bytes = input.inline_bytes;
                if (bytes.empty()) {
                    error_at(i, "the inline input is empty");
                    ok = false;
                    break;
                }
                CommandList list;
                if (byte_addressable) {
                    if (bytes.size() > resource.byte_size) {
                        error_at(i, luisa::format("the inline input is {} bytes, which exceeds the {} "
                                                  "bytes of the resource",
                                                  bytes.size(), resource.byte_size));
                        ok = false;
                        break;
                    }
                    if (resource.stride > 1u && bytes.size() % resource.stride != 0u) {
                        error_at(i, luisa::format("the inline input is {} bytes, which is not a multiple "
                                                  "of the {}-byte element stride",
                                                  bytes.size(), resource.stride));
                        ok = false;
                        break;
                    }
                    list << luisa::make_unique<compute::BufferUploadCommand>(
                        resource.handle, 0u, bytes.size(), bytes.data());
                } else if (texture_like) {
                    auto expected = compute::pixel_storage_size(resource.storage, resource.extent);
                    if (bytes.size() != expected) {
                        error_at(i, luisa::format("the inline input is {} bytes, but the whole '{}' image "
                                                  "is {} bytes",
                                                  bytes.size(), pixel_storage_name(resource.storage),
                                                  expected));
                        ok = false;
                        break;
                    }
                    list << luisa::make_unique<compute::TextureUploadCommand>(
                        resource.handle, resource.storage, 0u, resource.extent,
                        bytes.data(), uint3{0u, 0u, 0u});
                } else {
                    error_at(i, "an inline input can only fill a buffer, an image or a volume");
                    ok = false;
                    break;
                }
                stream << list.commit() << compute::synchronize();
                break;
            }
            case InputJson::Kind::Resource: {
                auto source_index = index_of(input.resource);
                if (source_index == count) {
                    error_at(i, luisa::format("the input names the undefined resource '{}'",
                                              input.resource));
                    ok = false;
                    break;
                }
                if (!byte_addressable) {
                    error_at(i, "only a buffer, an image or a volume can be filled from another resource");
                    ok = false;
                    break;
                }
                auto &&source = _entries[source_index];
                if (source.spec.type != ResourceType::Buffer) {
                    error_at(i, luisa::format("the input source '{}' is not a buffer; an image or a "
                                              "volume cannot be a copy source",
                                              input.resource));
                    ok = false;
                    break;
                }
                if (source.spec.input.kind != InputJson::Kind::None &&
                    state[source_index] != kResourceLoaded) {
                    error_at(i, luisa::format("the input source '{}' is filled after '{}'; list it "
                                              "earlier in the document",
                                              input.resource, entry.spec.name));
                    ok = false;
                    break;
                }
                size_t src_offset = 0u;
                size_t size = 0u;
                luisa::string error;
                if (!resolve_resource_region(input.offset, input.size,
                                             source.resource.byte_size,
                                             resource.byte_size, src_offset, size,
                                             luisa::format("the resource '{}'", input.resource),
                                             error)) {
                    error_at(i, error);
                    ok = false;
                    break;
                }
                CommandList list;
                list << luisa::make_unique<compute::BufferCopyCommand>(
                    source.resource.handle, resource.handle, src_offset, 0u, size);
                stream << list.commit() << compute::synchronize();
                break;
            }
            case InputJson::Kind::File: {
                auto path = paths.resolve_input(input.file);
                size_t file_offset = 0u;
                size_t size = 0u;
                luisa::string error;
                if (!resolve_file_region(path, input.offset, input.size, resource.byte_size,
                                         file_offset, size, error)) {
                    error_at(i, error);
                    ok = false;
                    break;
                }
                if (texture_like) {
                    auto whole = compute::pixel_storage_size(resource.storage, resource.extent);
                    if (size != whole) {
                        error_at(i, luisa::format("the input region is {} bytes, but filling the '{}' "
                                                  "image needs exactly the {} bytes of its level 0",
                                                  size, pixel_storage_name(resource.storage), whole));
                        ok = false;
                        break;
                    }
                }
                DStorageCompression compression{};
                auto compression_name = effective_compression(input, document.config);
                auto compression_supported = parse_dstorage_compression(compression_name, compression);
                if (choose_dstorage(document.config.dstorage.enabled, dstorage, dstorage_stream,
                                    compression_supported, compression_name, path, file_offset,
                                    size, device.backend_name(), document.config.strict,
                                    diagnostics)) {
                    auto file = dstorage->open_file(luisa::to_string(path));
                    if (file) {
                        auto view = file.view(file_offset, size);
                        auto command = build_dstorage_read(view, entry, 0u, 0u, resource.extent,
                                                           compression);
                        if (command) {
                            *dstorage_stream << std::move(command) << compute::synchronize();
                            break;
                        }
                        error_at(i, "internal error: this resource cannot be a DirectStorage destination");
                        ok = false;
                        break;
                    }
                    error_at(i, luisa::format("the DirectStorage extension cannot open '{}'",
                                              luisa::to_string(path)));
                    ok = false;
                    break;
                }
                auto *staging_block = staging.allocate(size);
                if (!read_file_region(path, file_offset, size, staging_block, error)) {
                    error_at(i, error);
                    ok = false;
                    break;
                }
                CommandList list;
                if (byte_addressable) {
                    list << luisa::make_unique<compute::BufferUploadCommand>(
                        resource.handle, 0u, size, staging_block);
                } else {
                    list << luisa::make_unique<compute::TextureUploadCommand>(
                        resource.handle, resource.storage, 0u, resource.extent,
                        staging_block, uint3{0u, 0u, 0u});
                }
                // The staging block is arena-owned, but an upload of a resource
                // is complete before the next one starts.
                stream << list.commit() << compute::synchronize();
                break;
            }
        }
        // The resource has had its chance; a failed load is reported already and
        // must not be retried, and a successful one may serve later copies.
        state[i] = kResourceLoaded;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// shaders
// ---------------------------------------------------------------------------

namespace {

// The three backends this example can compile native shaders for.
[[nodiscard]] bool is_native_shader_backend(luisa::string_view backend) noexcept {
    return backend == "dx" || backend == "vk" || backend == "cuda";
}

// The language/backend matrix: dx -> HLSL only, vk -> HLSL or GLSL,
// cuda -> CUDA C++ (cuda_nvrtc) only.
[[nodiscard]] bool backend_supports_language(luisa::string_view backend,
                                             NativeShaderLanguage language) noexcept {
    if (backend == "dx") { return language == NativeShaderLanguage::HLSL; }
    if (backend == "vk") {
        return language == NativeShaderLanguage::HLSL ||
               language == NativeShaderLanguage::GLSL;
    }
    if (backend == "cuda") { return language == NativeShaderLanguage::CUDA_NVRTC; }
    return false;
}

// The language of a shader that does not declare one: the source file's
// extension, else the document's default.
[[nodiscard]] NativeShaderLanguage language_from_path(
    luisa::string_view path, NativeShaderLanguage fallback) noexcept {
    auto dot = path.rfind('.');
    if (dot == luisa::string_view::npos) { return fallback; }
    auto extension = path.substr(dot);
    if (enum_name_is(extension, ".hlsl")) { return NativeShaderLanguage::HLSL; }
    if (enum_name_is(extension, ".glsl")) { return NativeShaderLanguage::GLSL; }
    if (enum_name_is(extension, ".cuda") || enum_name_is(extension, ".cu")) {
        return NativeShaderLanguage::CUDA_NVRTC;
    }
    return fallback;
}

}// namespace

bool ShaderRegistry::compile_all(const DispatchJson &document,
                                 luisa::span<const ShaderJson> cli_shaders,
                                 luisa::string_view backend,
                                 NativeShaderExt *ext,
                                 const PathResolver &paths,
                                 Diagnostics &diagnostics) noexcept {
    _ext = ext;
    _native.clear();
    // Merge: the document's shaders first, then the command line's. A CLI
    // shader replaces *every* document entry of the same name, so an override
    // never leaves a variant of the same name behind.
    luisa::vector<const ShaderJson *> shaders;
    shaders.reserve(document.shaders.size() + cli_shaders.size());
    for (auto &&shader : document.shaders) { shaders.emplace_back(&shader); }
    for (auto &&shader : cli_shaders) {
        auto replaced = false;
        for (auto iter = shaders.begin(); iter != shaders.end();) {
            if ((*iter)->name == shader.name) {
                iter = shaders.erase(iter);
                replaced = true;
            } else {
                ++iter;
            }
        }
        shaders.emplace_back(&shader);
        if (replaced) {
            auto message = luisa::format(
                "shaders ({}): the command line shader replaces the document's entry "
                "of the same name",
                shader.name);
            if (document.config.strict) {
                diagnostics.error(std::move(message));
            } else {
                diagnostics.warning(std::move(message));
            }
        }
    }
    if (shaders.empty()) { return true; }
    if (!is_native_shader_backend(backend)) {
        // A backend outside the example's native-shader scope: report once and
        // let the DSL-only parts of the document run.
        LUISA_WARNING("backend '{}' compiles no native shaders (dx: HLSL only; "
                      "vk: HLSL or GLSL; cuda: CUDA C++ (cuda_nvrtc) only): the "
                      "{} shader(s) of this document are skipped",
                      backend, shaders.size());
        return true;
    }
    if (ext == nullptr) {
        // The matrix is still validated below: a document that asks for a
        // language this backend cannot compile is a document error even when the
        // backend has no native shader support at all.
        LUISA_INFO("backend '{}' has no NativeShaderExt: the {} native shader(s) of "
                   "this document are skipped",
                   backend, shaders.size());
    }
    auto ok = true;
    // The language of a shader that omits both the key and a known extension.
    auto default_language = NativeShaderLanguage::HLSL;
    if (!document.config.default_language.empty() &&
        !parse_native_shader_language(document.config.default_language, default_language)) {
        diagnostics.error(luisa::format("config: unknown default_language '{}' "
                                        "(hlsl | glsl | cuda_nvrtc)",
                                        document.config.default_language));
        ok = false;
    }
    // A name may be declared once per language (see the README): the backend
    // picks the variant it speaks, in this preference order.
    auto preferred_languages = luisa::vector<NativeShaderLanguage>{};
    if (backend == "dx") {
        preferred_languages = {NativeShaderLanguage::HLSL};
    } else if (backend == "vk") {
        preferred_languages = {NativeShaderLanguage::GLSL, NativeShaderLanguage::HLSL};
    } else if (backend == "cuda") {
        preferred_languages = {NativeShaderLanguage::CUDA_NVRTC};
    }
    auto language_rank = [&](NativeShaderLanguage language) noexcept {
        for (auto i = size_t{0u}; i < preferred_languages.size(); i++) {
            if (preferred_languages[i] == language) { return i; }
        }
        return preferred_languages.size();
    };
    auto selected = luisa::vector<const ShaderJson *>{};
    auto selected_languages = luisa::vector<NativeShaderLanguage>{};
    for (auto *shader : shaders) {
        auto language = shader->language;
        if (!shader->has_language) {
            language = language_from_path(shader->path, default_language);
        }
        auto index = selected.size();
        for (auto i = size_t{0u}; i < selected.size(); i++) {
            if (selected[i]->name == shader->name) {
                index = i;
                break;
            }
        }
        if (index == selected.size()) {
            selected.emplace_back(shader);
            selected_languages.emplace_back(language);
        } else if (language_rank(language) < language_rank(selected_languages[index])) {
            LUISA_INFO("shader '{}': using the {} variant",
                       shader->name, native_shader_language_name(language));
            selected[index] = shader;
            selected_languages[index] = language;
        } else {
            LUISA_INFO("shader '{}': skipping the {} variant (this backend uses {})",
                       shader->name, native_shader_language_name(language),
                       native_shader_language_name(selected_languages[index]));
        }
    }
    for (auto i = size_t{0u}; i < selected.size(); i++) {
        auto *shader = selected[i];
        auto language = selected_languages[i];
        if (!backend_supports_language(backend, language)) {
            diagnostics.error(luisa::format(
                "shaders ({}): backend '{}' cannot compile a native {} shader "
                "(dx: HLSL only; vk: HLSL or GLSL; cuda: CUDA C++ (cuda_nvrtc) only)",
                shader->name, backend, native_shader_language_name(language)));
            ok = false;
            continue;
        }
        if (ext == nullptr) { continue; }
        NativeShaderCompileInfo info;
        info.language = language;
        // `info.source` is a string view: the path string must outlive the
        // compile call.
        luisa::string source_path;
        if (!shader->path.empty()) {
            auto resolved = paths.resolve_shader(shader->path);
            source_path = luisa::to_string(resolved);
            if (source_path.empty()) {
                diagnostics.error(luisa::format(
                    "shaders ({}): the source path '{}' is not a decodable path",
                    shader->name, shader->path));
                ok = false;
                continue;
            }
            info.source_type = NativeShaderSourceType::FilePath;
            info.source = source_path;
        } else if (!shader->source.empty()) {
            info.source_type = NativeShaderSourceType::SourceCode;
            info.source = shader->source;
        } else {
            diagnostics.error(luisa::format(
                "shaders ({}): the shader has neither a source file nor inline source",
                shader->name));
            ok = false;
            continue;
        }
        info.entry_point = shader->entry_point.empty() ?
                               luisa::string_view{"main"} :
                               luisa::string_view{shader->entry_point};
        info.shader_model = document.config.shader_model > 0u ?
                                document.config.shader_model :
                                65u;
        auto block_size = shader->block_size;
        if (block_size.x == 0u || block_size.y == 0u || block_size.z == 0u) {
            block_size = document.config.block_size;
        }
        info.block_size = block_size;
        info.push_constant_size = shader->push_constant_size > 0u ?
                                      shader->push_constant_size :
                                      (document.config.has_push_constant_size ?
                                           document.config.push_constant_size :
                                           0u);
        info.optimize = shader->optimize;
        info.enable_fast_math = shader->fast_math;
        info.enable_debug_info = shader->debug_info;
        for (auto &&directory : shader->include_dirs) {
            info.include_dirs.emplace_back(paths.resolve_shader(directory));
        }
        auto result = ext->compile(info);
        if (!result.ok()) {
            diagnostics.error(luisa::format(
                "shaders ({}): {}", shader->name,
                result.error.empty() ?
                    luisa::string{"the native shader compiler produced no binary"} :
                    result.error));
            ok = false;
            continue;
        }
        LUISA_INFO("compiled a native {} shader '{}': {} bytes, workgroup size "
                   "({} {} {}), {} reflected binding(s)",
                   native_shader_language_name(result.language), shader->name,
                   result.binary.size(), result.block_size.x, result.block_size.y,
                   result.block_size.z, result.bindings.size());
        for (auto binding_index = size_t{0u}; binding_index < result.bindings.size(); binding_index++) {
            auto &&binding = result.bindings[binding_index];
            LUISA_INFO("  binding {}: set {} register {} array {} usage {}",
                       binding_index, binding.space_index, binding.register_index,
                       binding.array_size, usage_name(binding.usage));
        }
        auto metadata = ext->load(result);
        if (!metadata.valid()) {
            diagnostics.error(luisa::format(
                "shaders ({}): the backend rejected the compiled {} shader and "
                "created no shader instance",
                shader->name, native_shader_language_name(language)));
            ok = false;
            continue;
        }
        _native.emplace_back(NativeEntry{shader->name,
                                         NativeShader{*ext, std::move(metadata)},
                                         source_path});
    }
    return ok;
}

void ShaderRegistry::add_dsl(luisa::string name, Owner owner, uint64_t handle,
                             size_t argument_count, size_t uniform_size,
                             uint3 block_size, uint32_t dimension) noexcept {
    auto entry = DslEntry{std::move(name), std::move(owner), handle,
                          argument_count, uniform_size, block_size, dimension};
    // Re-registering a name replaces the previous entry, so that a kernel can be
    // recompiled (the interactive display kernel is created once per session).
    for (auto &&existing : _dsl) {
        if (existing.name == entry.name) {
            existing = std::move(entry);
            return;
        }
    }
    _dsl.emplace_back(std::move(entry));
}

const ShaderRegistry::NativeEntry *ShaderRegistry::find_native(luisa::string_view name) const noexcept {
    for (auto &&entry : _native) {
        if (entry.name == name) { return &entry; }
    }
    return nullptr;
}

const ShaderRegistry::DslEntry *ShaderRegistry::find_dsl(luisa::string_view name) const noexcept {
    for (auto &&entry : _dsl) {
        if (entry.name == name) { return &entry; }
    }
    return nullptr;
}

bool ShaderRegistry::has_shader(luisa::string_view name) const noexcept {
    return find_native(name) != nullptr || find_dsl(name) != nullptr;
}

// ---------------------------------------------------------------------------
// workflow command builders (the commands without cross-stream state)
// ---------------------------------------------------------------------------

namespace {

void report_command_error(size_t index, const CommandJson &command,
                          luisa::string message,
                          Diagnostics &diagnostics) noexcept {
    diagnostics.error(luisa::format("workflow[{}] ({}): {}", index,
                                    command_kind_name(command.kind),
                                    std::move(message)));
}

// A resource of the expected kind, or null after reporting the problem.
[[nodiscard]] const ResourceRegistry::Entry *require_resource(
    const ResourceRegistry &resources, luisa::string_view name, ResourceType type,
    size_t index, const CommandJson &command, Diagnostics &diagnostics) noexcept {
    auto *entry = resources.find(name);
    if (entry == nullptr) {
        report_command_error(index, command,
                             luisa::format("the document defines no resource named '{}'",
                                           name),
                             diagnostics);
        return nullptr;
    }
    if (entry->spec.type != type) {
        report_command_error(index, command,
                             luisa::format("'{}' is a {}, but a {} is required here",
                                           name, to_string(entry->spec.type),
                                           to_string(type)),
                             diagnostics);
        return nullptr;
    }
    return entry;
}

// A buffer resource, or null after reporting the problem.
[[nodiscard]] const ResourceRegistry::Entry *require_buffer(
    const ResourceRegistry &resources, luisa::string_view name, size_t index,
    const CommandJson &command, Diagnostics &diagnostics) noexcept {
    auto *entry = resources.find(name);
    if (entry == nullptr) {
        report_command_error(index, command,
                             luisa::format("the document defines no resource named '{}'",
                                           name),
                             diagnostics);
        return nullptr;
    }
    if (entry->spec.type != ResourceType::Buffer) {
        report_command_error(index, command,
                             luisa::format("'{}' is a {}, not a buffer",
                                           name, to_string(entry->spec.type)),
                             diagnostics);
        return nullptr;
    }
    return entry;
}

// Resolves a byte range inside a buffer: a requested size of 0 is "the rest of
// the buffer from `offset`".
[[nodiscard]] bool resolve_buffer_range(const ResourceRegistry::Entry &entry,
                                        size_t offset, size_t size,
                                        size_t &resolved_offset,
                                        size_t &resolved_size,
                                        luisa::string &error) noexcept {
    return resolve_resource_region(offset, size, entry.resource.byte_size,
                                   entry.resource.byte_size, resolved_offset,
                                   resolved_size,
                                   luisa::format("the buffer '{}'", entry.spec.name),
                                   error);
}

// The build request of a build command ("prefer_update" | "force_build").
[[nodiscard]] bool parse_build_request(const CommandJson &command, size_t index,
                                       AccelBuildRequest &request,
                                       Diagnostics &diagnostics) noexcept {
    if (command.request.empty()) {
        request = AccelBuildRequest::PREFER_UPDATE;
        return true;
    }
    if (parse_accel_request(command.request, request)) { return true; }
    report_command_error(index, command,
                         luisa::format("unknown build request '{}'", command.request),
                         diagnostics);
    return false;
}

[[nodiscard]] bool append_native_dispatch(const CommandJson &command, size_t index,
                                          const ResourceRegistry &resources,
                                          const ShaderRegistry &shaders,
                                          CommandList &list, size_t &command_count,
                                          Diagnostics &diagnostics) noexcept {
    auto *shader = shaders.find_native(command.shader);
    if (shader == nullptr) {
        report_command_error(index, command,
                             luisa::format("'{}' is not a compiled native shader",
                                           command.shader),
                             diagnostics);
        return false;
    }
    auto launcher = shader->shader.launcher();
    launcher.set_allow_usage_override(command.allow_usage_override);
    auto ok = true;
    for (auto &&binding : command.bindings) {
        auto *entry = require_buffer(resources, binding.resource, index, command, diagnostics);
        if (entry == nullptr) {
            ok = false;
            continue;
        }
        Usage usage{};
        if (!parse_usage(binding.usage, usage)) {
            report_command_error(index, command,
                                 luisa::format("unknown usage '{}' for the binding of '{}'",
                                               binding.usage, binding.resource),
                                 diagnostics);
            ok = false;
            continue;
        }
        if (usage == Usage::NONE) {
            report_command_error(index, command,
                                 luisa::format("the binding of '{}' declares no usage",
                                               binding.resource),
                                 diagnostics);
            ok = false;
            continue;
        }
        size_t offset = 0u;
        size_t size = 0u;
        luisa::string error;
        if (!resolve_buffer_range(*entry, binding.offset, binding.size, offset, size, error)) {
            report_command_error(index, command, std::move(error), diagnostics);
            ok = false;
            continue;
        }
        if (size == 0u) {
            report_command_error(index, command,
                                 luisa::format("the binding of '{}' is empty",
                                               binding.resource),
                                 diagnostics);
            ok = false;
            continue;
        }
        if (binding.has_index) {
            // Reflection index: the only unambiguous selector for a DirectX
            // shader whose HLSL register namespaces collide.
            launcher.add_buffer_by_index(binding.index, entry->resource.handle,
                                         offset, size, usage);
        } else if (binding.has_register) {
            launcher.add_buffer(binding.reg, binding.space, entry->resource.handle,
                                offset, size, usage);
        } else {
            // Neither form was given: the positional form, which fills the
            // canonical (space, register) order of the reflection table.
            launcher.add_buffer(entry->resource.handle, offset, size, usage);
        }
    }
    for (auto &&uniform : command.uniforms) {
        if (uniform.bytes.empty()) {
            report_command_error(index, command, "a uniform has no payload", diagnostics);
            ok = false;
            continue;
        }
        if (uniform.alignment == 0u || uniform.alignment > 16u) {
            report_command_error(index, command,
                                 luisa::format("the uniform alignment {} is outside [1, 16]",
                                               uniform.alignment),
                                 diagnostics);
            ok = false;
            continue;
        }
        launcher.add_uniform(uniform.bytes.data(), uniform.bytes.size(),
                             uniform.alignment);
    }
    if (!ok) { return false; }
    // `validate()` reports exactly what `build()` asserts, so a bad binding or
    // uniform is a diagnostic instead of an abort.
    if (auto error = launcher.validate(); !error.empty()) {
        diagnostics.error(luisa::format("workflow[{}] ({}): {}", index,
                                        command_kind_name(command.kind), error));
        return false;
    }
    auto block_size = shader->shader.block_size();
    if (block_size.x == 0u || block_size.y == 0u || block_size.z == 0u) {
        report_command_error(index, command,
                             "the compiled shader reflects no workgroup size",
                             diagnostics);
        return false;
    }
    auto thread_count = command.dispatch;
    if (thread_count.x == 0u || thread_count.y == 0u || thread_count.z == 0u) {
        auto grid = command.grid;
        if (grid.x == 0u || grid.y == 0u || grid.z == 0u) {
            report_command_error(index, command,
                                 "a native dispatch needs either a nonzero 'dispatch' "
                                 "(threads) or a nonzero 'grid' (thread groups)",
                                 diagnostics);
            return false;
        }
        thread_count = grid * block_size;
    }
    list << std::move(launcher).build(thread_count);
    command_count++;
    return true;
}

[[nodiscard]] bool append_shader_dispatch(const CommandJson &command, size_t index,
                                          const ResourceRegistry &resources,
                                          const ShaderRegistry &shaders,
                                          CommandList &list, size_t &command_count,
                                          Diagnostics &diagnostics) noexcept {
    auto *shader = shaders.find_dsl(command.shader);
    if (shader == nullptr) {
        report_command_error(index, command,
                             luisa::format("'{}' is not a registered DSL shader",
                                           command.shader),
                             diagnostics);
        return false;
    }
    if (command.arguments.size() != shader->argument_count) {
        report_command_error(index, command,
                             luisa::format("the shader '{}' takes {} argument(s), but the "
                                           "document provides {}; the encoder requires one "
                                           "argument per shader parameter",
                                           command.shader, shader->argument_count,
                                           command.arguments.size()),
                             diagnostics);
        return false;
    }
    ComputeDispatchCmdEncoder encoder{shader->handle, shader->argument_count,
                                      shader->uniform_size};
    auto ok = true;
    for (auto &&argument : command.arguments) {
        if (enum_name_is(argument.kind, "buffer")) {
            auto *entry = require_resource(resources, argument.resource, ResourceType::Buffer,
                                           index, command, diagnostics);
            if (entry == nullptr) {
                ok = false;
                continue;
            }
            size_t offset = 0u;
            size_t size = 0u;
            luisa::string error;
            if (!resolve_buffer_range(*entry, argument.offset, 0u, offset, size, error)) {
                report_command_error(index, command, std::move(error), diagnostics);
                ok = false;
                continue;
            }
            encoder.encode_buffer(entry->resource.handle, offset, size);
        } else if (enum_name_is(argument.kind, "texture")) {
            auto *entry = resources.find(argument.resource);
            if (entry == nullptr ||
                (entry->spec.type != ResourceType::Texture &&
                 entry->spec.type != ResourceType::Volume)) {
                report_command_error(index, command,
                                     luisa::format("'{}' is not an image or volume resource",
                                                   argument.resource),
                                     diagnostics);
                ok = false;
                continue;
            }
            encoder.encode_texture(entry->resource.handle, argument.level);
        } else if (enum_name_is(argument.kind, "bindless_array")) {
            auto *entry = require_resource(resources, argument.resource,
                                           ResourceType::BindlessArray, index, command,
                                           diagnostics);
            if (entry == nullptr) {
                ok = false;
                continue;
            }
            encoder.encode_bindless_array(entry->resource.handle);
        } else if (enum_name_is(argument.kind, "accel")) {
            auto *entry = require_resource(resources, argument.resource,
                                           ResourceType::Accel, index, command, diagnostics);
            if (entry == nullptr) {
                ok = false;
                continue;
            }
            encoder.encode_accel(entry->resource.handle);
        } else if (enum_name_is(argument.kind, "uniform")) {
            if (argument.bytes.empty()) {
                report_command_error(index, command, "a uniform argument has no payload",
                                     diagnostics);
                ok = false;
                continue;
            }
            if (argument.alignment == 0u || argument.alignment > 16u) {
                report_command_error(index, command,
                                     luisa::format("the uniform alignment {} is outside [1, 16]",
                                                   argument.alignment),
                                     diagnostics);
                ok = false;
                continue;
            }
            encoder.encode_uniform(argument.bytes.data(), argument.bytes.size(),
                                   argument.alignment);
        } else {
            report_command_error(index, command,
                                 luisa::format("unknown argument kind '{}'", argument.kind),
                                 diagnostics);
            ok = false;
            continue;
        }
    }
    if (!ok) { return false; }
    if (!command.batched.empty()) {
        for (auto &&size : command.batched) {
            if (size.x == 0u || size.y == 0u || size.z == 0u) {
                report_command_error(index, command,
                                     "a batched dispatch size must be nonzero in every "
                                     "dimension",
                                     diagnostics);
                return false;
            }
        }
        encoder.set_dispatch_sizes(command.batched);
    } else {
        auto dispatch = command.dispatch;
        if (dispatch.x == 0u || dispatch.y == 0u || dispatch.z == 0u) {
            report_command_error(index, command,
                                 "a shader dispatch needs a nonzero 'dispatch' size, or a "
                                 "nonempty 'batched' array of sizes",
                                 diagnostics);
            return false;
        }
        encoder.set_dispatch_size(dispatch);
    }
    list << std::move(encoder).build();
    command_count++;
    return true;
}

[[nodiscard]] bool append_bindless_array_update(const CommandJson &command, size_t index,
                                                const ResourceRegistry &resources,
                                                CommandList &list, size_t &command_count,
                                                Diagnostics &diagnostics) noexcept {
    using BA = BindlessArrayUpdateCommand;
    auto *array = require_resource(resources, command.resource,
                                   ResourceType::BindlessArray, index, command, diagnostics);
    if (array == nullptr) { return false; }
    auto slot_count = array->spec.slot_count;
    auto operation_of = [](luisa::string_view op, BA::Operation &operation) noexcept {
        if (op.empty() || enum_name_is(op, "emplace")) {
            operation = BA::Operation::EMPLACE;
            return true;
        }
        if (enum_name_is(op, "remove")) {
            operation = BA::Operation::REMOVE;
            return true;
        }
        return false;
    };
    auto ok = true;
    auto check_slot = [&](const BindlessModJson &mod) noexcept {
        if (mod.slot < slot_count) { return true; }
        report_command_error(index, command,
                             luisa::format("slot {} of '{}' is outside the {} slots of the "
                                           "bindless array",
                                           mod.slot, command.resource, slot_count),
                             diagnostics);
        return false;
    };
    auto fill_buffer = [&](const BindlessModJson &mod, BA::ModifiedBuffer &out) noexcept {
        BA::Operation operation{};
        if (!operation_of(mod.op, operation)) {
            report_command_error(index, command,
                                 luisa::format("unknown bindless operation '{}'", mod.op),
                                 diagnostics);
            return false;
        }
        if (operation == BA::Operation::REMOVE) {
            out = BA::ModifiedBuffer::remove();
            return true;
        }
        auto *entry = require_buffer(resources, mod.resource, index, command, diagnostics);
        if (entry == nullptr) { return false; }
        auto size = mod.size == 0u ? BA::ModifiedBuffer::whole_buffer_size : mod.size;
        if (mod.offset > entry->resource.byte_size ||
            (size != BA::ModifiedBuffer::whole_buffer_size &&
             size > entry->resource.byte_size - mod.offset)) {
            report_command_error(index, command,
                                 luisa::format("the slot region [{}, {}) of '{}' exceeds its "
                                               "{} bytes",
                                               mod.offset,
                                               size == BA::ModifiedBuffer::whole_buffer_size ?
                                                   entry->resource.byte_size :
                                                   mod.offset + size,
                                               mod.resource, entry->resource.byte_size),
                                 diagnostics);
            return false;
        }
        out = BA::ModifiedBuffer::emplace(entry->resource.handle, mod.offset, size);
        return true;
    };
    auto fill_texture = [&](const BindlessModJson &mod, BA::ModifiedTexture &out,
                            ResourceType type) noexcept {
        BA::Operation operation{};
        if (!operation_of(mod.op, operation)) {
            report_command_error(index, command,
                                 luisa::format("unknown bindless operation '{}'", mod.op),
                                 diagnostics);
            return false;
        }
        if (operation == BA::Operation::REMOVE) {
            out = BA::ModifiedTexture::remove();
            return true;
        }
        auto *entry = require_resource(resources, mod.resource, type, index, command,
                                       diagnostics);
        if (entry == nullptr) { return false; }
        Sampler sampler{};
        if (!parse_sampler(mod.sampler, sampler)) {
            report_command_error(index, command,
                                 luisa::format("unknown sampler (filter '{}', address '{}')",
                                               mod.sampler.filter, mod.sampler.address),
                                 diagnostics);
            return false;
        }
        out = BA::ModifiedTexture::emplace(entry->resource.handle, sampler);
        return true;
    };
    auto mode = command.mode.empty() ? luisa::string_view{"multiple"} :
                                       luisa::string_view{command.mode};
    if (enum_name_is(mode, "multiple")) {
        luisa::vector<BA::Modification> modifications;
        modifications.reserve(command.modifications.size());
        for (auto &&mod : command.modifications) {
            if (!check_slot(mod)) {
                ok = false;
                continue;
            }
            BA::Modification modification{mod.slot};
            if (mod.kind.empty() || enum_name_is(mod.kind, "buffer")) {
                if (!fill_buffer(mod, modification.buffer)) {
                    ok = false;
                    continue;
                }
            } else if (enum_name_is(mod.kind, "texture2d")) {
                if (!fill_texture(mod, modification.tex2d, ResourceType::Texture)) {
                    ok = false;
                    continue;
                }
            } else if (enum_name_is(mod.kind, "texture3d")) {
                if (!fill_texture(mod, modification.tex3d, ResourceType::Volume)) {
                    ok = false;
                    continue;
                }
            } else {
                report_command_error(index, command,
                                     luisa::format("unknown bindless slot kind '{}'", mod.kind),
                                     diagnostics);
                ok = false;
                continue;
            }
            modifications.emplace_back(modification);
        }
        if (!ok) { return false; }
        list << luisa::make_unique<BA>(array->resource.handle, std::move(modifications));
    } else if (enum_name_is(mode, "buffer")) {
        luisa::vector<BA::BufferModification> modifications;
        modifications.reserve(command.modifications.size());
        for (auto &&mod : command.modifications) {
            if (!check_slot(mod)) {
                ok = false;
                continue;
            }
            BA::BufferModification modification{mod.slot};
            if (!fill_buffer(mod, modification.buffer)) {
                ok = false;
                continue;
            }
            modifications.emplace_back(modification);
        }
        if (!ok) { return false; }
        list << luisa::make_unique<BA>(array->resource.handle, std::move(modifications));
    } else if (enum_name_is(mode, "texture2d") || enum_name_is(mode, "texture3d")) {
        auto type = enum_name_is(mode, "texture2d") ? ResourceType::Texture :
                                                      ResourceType::Volume;
        if (type == ResourceType::Texture) {
            luisa::vector<BA::Texture2DModification> modifications;
            modifications.reserve(command.modifications.size());
            for (auto &&mod : command.modifications) {
                if (!check_slot(mod)) {
                    ok = false;
                    continue;
                }
                BA::Texture2DModification modification{mod.slot};
                if (!fill_texture(mod, modification.tex2d, type)) {
                    ok = false;
                    continue;
                }
                modifications.emplace_back(modification);
            }
            if (!ok) { return false; }
            list << luisa::make_unique<BA>(array->resource.handle, std::move(modifications));
        } else {
            luisa::vector<BA::Texture3DModification> modifications;
            modifications.reserve(command.modifications.size());
            for (auto &&mod : command.modifications) {
                if (!check_slot(mod)) {
                    ok = false;
                    continue;
                }
                BA::Texture3DModification modification{mod.slot};
                if (!fill_texture(mod, modification.tex3d, type)) {
                    ok = false;
                    continue;
                }
                modifications.emplace_back(modification);
            }
            if (!ok) { return false; }
            list << luisa::make_unique<BA>(array->resource.handle, std::move(modifications));
        }
    } else {
        report_command_error(index, command,
                             luisa::format("unknown bindless array mode '{}' "
                                           "(multiple | buffer | texture2d | texture3d)",
                                           command.mode),
                             diagnostics);
        return false;
    }
    command_count++;
    return true;
}

[[nodiscard]] bool append_mesh_build(const CommandJson &command, size_t index,
                                     const ResourceRegistry &resources,
                                     CommandList &list, size_t &command_count,
                                     Diagnostics &diagnostics) noexcept {
    auto *mesh = require_resource(resources, command.resource, ResourceType::Mesh,
                                  index, command, diagnostics);
    auto *vertices = require_buffer(resources, command.vertex_buffer, index, command, diagnostics);
    auto *triangles = require_buffer(resources, command.triangle_buffer, index, command, diagnostics);
    if (mesh == nullptr || vertices == nullptr || triangles == nullptr) { return false; }
    AccelBuildRequest request{};
    if (!parse_build_request(command, index, request, diagnostics)) { return false; }
    auto vertex_stride = command.vertex_stride != 0u ?
                             static_cast<size_t>(command.vertex_stride) :
                             vertices->resource.stride;
    if (vertex_stride == 0u) {
        report_command_error(index, command,
                             "the vertex buffer has no element stride; the mesh build needs "
                             "a 'vertex_stride' or a typed vertex buffer",
                             diagnostics);
        return false;
    }
    size_t vertex_offset = 0u;
    size_t vertex_size = 0u;
    size_t triangle_offset = 0u;
    size_t triangle_size = 0u;
    luisa::string error;
    if (!resolve_buffer_range(*vertices, command.vertex_buffer_offset,
                              command.vertex_buffer_size, vertex_offset, vertex_size,
                              error) ||
        !resolve_buffer_range(*triangles, command.triangle_buffer_offset,
                              command.triangle_buffer_size, triangle_offset,
                              triangle_size, error)) {
        report_command_error(index, command, std::move(error), diagnostics);
        return false;
    }
    if (vertex_size == 0u || vertex_size % vertex_stride != 0u) {
        report_command_error(index, command,
                             luisa::format("the {} byte vertex region is not a nonzero "
                                           "multiple of the {}-byte stride",
                                           vertex_size, vertex_stride),
                             diagnostics);
        return false;
    }
    if (triangle_size == 0u || triangle_size % sizeof(compute::Triangle) != 0u) {
        report_command_error(index, command,
                             luisa::format("the {} byte triangle region is not a nonzero "
                                           "multiple of a {}-byte triangle",
                                           triangle_size, sizeof(compute::Triangle)),
                             diagnostics);
        return false;
    }
    list << luisa::make_unique<MeshBuildCommand>(
        mesh->resource.handle, request, vertices->resource.handle, vertex_offset,
        vertex_size, vertex_stride, triangles->resource.handle, triangle_offset,
        triangle_size);
    command_count++;
    return true;
}

[[nodiscard]] bool append_procedural_primitive_build(const CommandJson &command, size_t index,
                                                     const ResourceRegistry &resources,
                                                     CommandList &list, size_t &command_count,
                                                     Diagnostics &diagnostics) noexcept {
    auto *primitive = require_resource(resources, command.resource,
                                       ResourceType::ProceduralPrimitive, index, command,
                                       diagnostics);
    auto *boxes = require_buffer(resources, command.aabb_buffer, index, command, diagnostics);
    if (primitive == nullptr || boxes == nullptr) { return false; }
    AccelBuildRequest request{};
    if (!parse_build_request(command, index, request, diagnostics)) { return false; }
    size_t box_offset = 0u;
    size_t box_size = 0u;
    luisa::string error;
    if (!resolve_buffer_range(*boxes, command.aabb_buffer_offset, command.aabb_buffer_size,
                              box_offset, box_size, error)) {
        report_command_error(index, command, std::move(error), diagnostics);
        return false;
    }
    if (box_size == 0u || box_size % sizeof(compute::AABB) != 0u) {
        report_command_error(index, command,
                             luisa::format("the {} byte AABB region is not a nonzero multiple "
                                           "of a {}-byte AABB",
                                           box_size, sizeof(compute::AABB)),
                             diagnostics);
        return false;
    }
    list << luisa::make_unique<ProceduralPrimitiveBuildCommand>(
        primitive->resource.handle, request, boxes->resource.handle, box_offset, box_size);
    command_count++;
    return true;
}

[[nodiscard]] bool append_accel_build(const CommandJson &command, size_t index,
                                      const ResourceRegistry &resources,
                                      CommandList &list, size_t &command_count,
                                      Diagnostics &diagnostics) noexcept {
    auto *accel = require_resource(resources, command.resource, ResourceType::Accel,
                                   index, command, diagnostics);
    if (accel == nullptr) { return false; }
    AccelBuildRequest request{};
    if (!parse_build_request(command, index, request, diagnostics)) { return false; }
    if (command.update_instance_buffer_only && !command.accel_modifications.empty()) {
        report_command_error(index, command,
                             "an instance-buffer-only update cannot carry modifications",
                             diagnostics);
        return false;
    }
    luisa::vector<AccelBuildCommand::Modification> modifications;
    modifications.reserve(command.accel_modifications.size());
    auto instance_count = command.instance_count;
    for (auto &&mod : command.accel_modifications) {
        AccelBuildCommand::Modification modification{mod.index};
        if (mod.has_user_id) { modification.set_user_id(mod.user_id); }
        if (mod.has_visibility) {
            modification.set_visibility(static_cast<uint8_t>(mod.visibility));
        }
        if (mod.has_opaque) { modification.set_opaque(mod.opaque); }
        if (mod.has_transform) {
            // The document writes the transform row-major with the translation
            // in the fourth column, which is exactly the layout
            // `set_transform_data` expects.
            modification.set_transform_data(mod.transform);
        }
        if (mod.has_primitive) {
            auto *primitive = resources.find(mod.primitive);
            if (primitive == nullptr ||
                (primitive->spec.type != ResourceType::Mesh &&
                 primitive->spec.type != ResourceType::ProceduralPrimitive)) {
                report_command_error(index, command,
                                     luisa::format("'{}' is not a mesh or procedural primitive "
                                                   "resource",
                                                   mod.primitive),
                                     diagnostics);
                return false;
            }
            modification.set_primitive(primitive->resource.handle);
        }
        instance_count = std::max(instance_count, mod.index + 1u);
        modifications.emplace_back(modification);
    }
    list << luisa::make_unique<AccelBuildCommand>(
        accel->resource.handle, instance_count, request, std::move(modifications),
        command.update_instance_buffer_only);
    command_count++;
    return true;
}

}// namespace

// ---------------------------------------------------------------------------
// workflow execution
// ---------------------------------------------------------------------------

namespace {

// One download whose target lives in the arena until the frame is synchronized.
struct PendingDownload {
    size_t index{0u};
    const std::byte *data{nullptr};
    size_t size{0u};
    uint3 extent{0u, 0u, 0u};
    PixelStorage storage{PixelStorage::BYTE1};
    const OutputJson *output{nullptr};
    const VerifyJson *verify{nullptr};
    const std::byte *verify_source{nullptr};
};

[[nodiscard]] bool is_texture_like(const ResourceRegistry::Entry *entry) noexcept {
    return entry != nullptr &&
           (entry->spec.type == ResourceType::Texture ||
            entry->spec.type == ResourceType::Volume);
}

// Resolves the level, pixel storage and 3-D region of a texture/volume command.
// A requested size of 0 in any dimension means "the whole mip level"; the
// storage must agree with the resource it names.
[[nodiscard]] bool resolve_texture_region(const ResourceRegistry::Entry &entry,
                                          uint32_t level, luisa::string_view storage_name,
                                          uint3 requested_size, uint3 requested_offset,
                                          PixelStorage &storage, uint3 &size,
                                          luisa::string &error) noexcept {
    if (level >= entry.resource.levels) {
        error = luisa::format("mip level {} exceeds the {} level(s) of '{}'",
                              level, entry.resource.levels, entry.spec.name);
        return false;
    }
    if (!storage_name.empty()) {
        if (!parse_pixel_storage(storage_name, storage)) {
            error = luisa::format("unknown pixel storage '{}'", storage_name);
            return false;
        }
        if (storage != entry.resource.storage) {
            error = luisa::format("'{}' stores '{}' pixels, not '{}'",
                                  entry.spec.name,
                                  pixel_storage_name(entry.resource.storage),
                                  storage_name);
            return false;
        }
    } else {
        storage = entry.resource.storage;
    }
    auto full = mip_extent(entry.resource.extent, level);
    size = requested_size.x == 0u || requested_size.y == 0u || requested_size.z == 0u ?
               full :
               requested_size;
    if (size.x > full.x || size.y > full.y || size.z > full.z) {
        error = luisa::format("the region ({} {} {}) exceeds the ({} {} {}) pixels of mip {} of '{}'",
                              size.x, size.y, size.z, full.x, full.y, full.z, level,
                              entry.spec.name);
        return false;
    }
    if (requested_offset.x + size.x > full.x ||
        requested_offset.y + size.y > full.y ||
        requested_offset.z + size.z > full.z) {
        error = luisa::format("the region [{}, {}) [{}, {}) [{}, {}) exceeds the "
                              "({} {} {}) pixels of mip {} of '{}'",
                              requested_offset.x, requested_offset.x + size.x,
                              requested_offset.y, requested_offset.y + size.y,
                              requested_offset.z, requested_offset.z + size.z,
                              full.x, full.y, full.z, level, entry.spec.name);
        return false;
    }
    return true;
}

}// namespace

bool WorkflowExecutor::execute(const DispatchJson &document) noexcept {
    // The per-kind counters are rebuilt for every frame.
    std::fill(_command_counts.begin(), _command_counts.end(), size_t{0u});
    // Per-frame state. The frozen class declares no members for it, so one
    // `execute` owns one command segment, one payload arena and one batch of
    // downloads that is written and verified after the last synchronize.
    CommandList segment;
    ByteArena arena;
    luisa::vector<PendingDownload> downloads;
    TimelineEvent timeline_event;
    auto has_timeline_event = false;
    auto fence = uint64_t{0u};
    size_t command_count = 0u;
    auto ok = true;
    auto *dstorage = _device.extension<DStorageExt>();

    auto fail = [&](size_t index, const CommandJson &command,
                    luisa::string message) noexcept {
        report_command_error(index, command, std::move(message), _diagnostics);
    };
    auto flush = [&]() noexcept {
        // Committing an empty list is harmless and `commit()` moves the
        // commands out, so the segment stays usable afterwards.
        _stream << segment.commit();
    };
    // Hands a DirectStorage read over to the compute stream. The read runs on
    // the upload stream, so the segment queued so far is flushed first (it may
    // touch the same resource) and the compute stream waits on the timeline
    // event the upload signals; with `--sync-uploads` the upload stream is
    // synchronized instead and no event is used.
    auto submit_upload = [&](luisa::unique_ptr<compute::DStorageReadCommand> read) noexcept {
        flush();
        if (_sync_uploads) {
            *_upload_stream << std::move(read) << compute::synchronize();
            return;
        }
        if (!has_timeline_event) {
            timeline_event = _device.create_timeline_event();
            has_timeline_event = true;
        }
        ++fence;
        *_upload_stream << std::move(read) << timeline_event.signal(fence);
        _stream << timeline_event.wait(fence);
    };

    for (auto i = size_t{0u}; i < document.workflow.size(); i++) {
        auto &&command = document.workflow[i];
        switch (command.kind) {
            case CommandKind::Log: {
                LUISA_INFO("[workflow {}] {}", i, command.message);
                break;
            }
            case CommandKind::Synchronize: {
                Clock clock;
                flush();
                _stream.synchronize();
                if (!command.label.empty()) {
                    LUISA_INFO("[workflow {}] synchronized '{}' in {:.3f} ms",
                               i, command.label, clock.toc());
                }
                break;
            }
            case CommandKind::BufferUpload: {
                auto *destination = require_buffer(_resources, command.resource, i,
                                                   command, _diagnostics);
                if (destination == nullptr) {
                    ok = false;
                    break;
                }
                size_t destination_offset = 0u;
                size_t destination_size = 0u;
                luisa::string error;
                if (!resolve_buffer_range(*destination, command.offset, command.size,
                                          destination_offset, destination_size, error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                if (destination_size == 0u) {
                    fail(i, command, "the upload region is empty");
                    ok = false;
                    break;
                }
                auto &&input = command.input;
                switch (input.kind) {
                    case InputJson::Kind::None: {
                        fail(i, command, "a buffer upload needs an input");
                        ok = false;
                        break;
                    }
                    case InputJson::Kind::Inline: {
                        auto &&bytes = input.inline_bytes;
                        if (bytes.size() != destination_size) {
                            fail(i, command, luisa::format("the inline payload is {} bytes, but the "
                                                           "destination region is {} bytes",
                                                           bytes.size(), destination_size));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::BufferUploadCommand>(
                            destination->resource.handle, destination_offset,
                            destination_size, bytes.data());
                        command_count++;
                        break;
                    }
                    case InputJson::Kind::Resource: {
                        auto *source = require_buffer(_resources, input.resource, i,
                                                      command, _diagnostics);
                        if (source == nullptr) {
                            ok = false;
                            break;
                        }
                        size_t source_offset = 0u;
                        size_t size = 0u;
                        if (!resolve_resource_region(
                                input.offset, input.size, source->resource.byte_size,
                                destination_size, source_offset, size,
                                luisa::format("the buffer '{}'", input.resource), error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        if (size != destination_size) {
                            fail(i, command, luisa::format("the source region is {} bytes, but the "
                                                           "destination region is {} bytes",
                                                           size, destination_size));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::BufferCopyCommand>(
                            source->resource.handle, destination->resource.handle,
                            source_offset, destination_offset, destination_size);
                        command_count++;
                        break;
                    }
                    case InputJson::Kind::File: {
                        auto path = _paths.resolve_input(input.file);
                        size_t file_offset = 0u;
                        size_t size = 0u;
                        if (!resolve_file_region(path, input.offset, input.size,
                                                 destination_size, file_offset, size,
                                                 error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        if (size != destination_size) {
                            fail(i, command, luisa::format("the file region is {} bytes, but the "
                                                           "destination region is {} bytes",
                                                           size, destination_size));
                            ok = false;
                            break;
                        }
                        DStorageCompression compression{};
                        auto compression_name = effective_compression(input, document.config);
                        auto compression_supported = parse_dstorage_compression(
                            compression_name, compression);
                        if (choose_dstorage(document.config.dstorage.enabled, dstorage,
                                            _upload_stream, compression_supported,
                                            compression_name, path, file_offset, size,
                                            _device.backend_name(),
                                            document.config.strict, _diagnostics)) {
                            auto file = dstorage->open_file(luisa::to_string(path));
                            if (!file) {
                                fail(i, command, luisa::format("the DirectStorage extension cannot open '{}'", luisa::to_string(path)));
                                ok = false;
                                break;
                            }
                            BufferView<std::byte> view{
                                destination->resource.native_handle,
                                destination->resource.handle, 1u, destination_offset,
                                size, destination->resource.byte_size};
                            submit_upload(file.view(file_offset, size).copy_to(view, compression));
                            command_count++;
                            break;
                        }
                        auto *staging = arena.allocate(size);
                        if (!read_file_region(path, file_offset, size, staging, error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::BufferUploadCommand>(
                            destination->resource.handle, destination_offset,
                            destination_size, staging);
                        command_count++;
                        break;
                    }
                }
                break;
            }
            case CommandKind::BufferDownload: {
                auto *source = require_buffer(_resources, command.resource, i,
                                              command, _diagnostics);
                if (source == nullptr) {
                    ok = false;
                    break;
                }
                size_t offset = 0u;
                size_t size = 0u;
                luisa::string error;
                if (!resolve_buffer_range(*source, command.offset, command.size,
                                          offset, size, error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                if (size == 0u) {
                    fail(i, command, "the download region is empty");
                    ok = false;
                    break;
                }
                auto *target = arena.allocate(size);
                segment << luisa::make_unique<compute::BufferDownloadCommand>(
                    source->resource.handle, offset, size, target);
                command_count++;
                PendingDownload pending;
                pending.index = i;
                pending.data = target;
                pending.size = size;
                pending.output = &command.output;
                pending.verify = &command.verify;
                if (!enum_name_is(command.verify.kind, "none") &&
                    !command.verify.kind.empty()) {
                    if (!enum_name_is(command.verify.kind, "linear") &&
                        !enum_name_is(command.verify.kind, "copy")) {
                        fail(i, command, luisa::format("unknown verification '{}' (none | linear | copy)", command.verify.kind));
                        ok = false;
                    } else {
                        auto *verification_source = require_buffer(
                            _resources, command.verify.source, i, command, _diagnostics);
                        if (verification_source == nullptr) {
                            ok = false;
                        } else if (verification_source->resource.byte_size < size) {
                            fail(i, command, luisa::format("the verification source '{}' holds {} bytes, "
                                                           "which is less than the {} downloaded bytes",
                                                           command.verify.source, verification_source->resource.byte_size, size));
                            ok = false;
                        } else {
                            auto *block = arena.allocate(size);
                            segment << luisa::make_unique<compute::BufferDownloadCommand>(
                                verification_source->resource.handle, 0u, size, block);
                            command_count++;
                            pending.verify_source = block;
                        }
                    }
                }
                downloads.emplace_back(std::move(pending));
                break;
            }
            case CommandKind::BufferCopy: {
                auto *source = require_buffer(_resources, command.src, i, command, _diagnostics);
                auto *destination = require_buffer(_resources, command.dst, i, command,
                                                   _diagnostics);
                if (source == nullptr || destination == nullptr) {
                    ok = false;
                    break;
                }
                size_t source_offset = 0u;
                size_t destination_offset = 0u;
                auto source_available = command.src_offset <= source->resource.byte_size ?
                                            source->resource.byte_size - command.src_offset :
                                            0u;
                auto destination_available = command.dst_offset <= destination->resource.byte_size ?
                                                 destination->resource.byte_size - command.dst_offset :
                                                 0u;
                if (command.src_offset > source->resource.byte_size ||
                    command.dst_offset > destination->resource.byte_size) {
                    fail(i, command, luisa::format("the copy offsets ({}, {}) exceed the sizes ({}, {}) "
                                                   "of '{}' and '{}'",
                                                   command.src_offset, command.dst_offset, source->resource.byte_size, destination->resource.byte_size, command.src, command.dst));
                    ok = false;
                    break;
                }
                auto size = command.size == 0u ?
                                std::min(source_available, destination_available) :
                                command.size;
                if (size == 0u || size > source_available || size > destination_available) {
                    fail(i, command, luisa::format("the {} byte copy does not fit the {} and {} bytes "
                                                   "available in '{}' and '{}'",
                                                   size, source_available, destination_available, command.src, command.dst));
                    ok = false;
                    break;
                }
                source_offset = command.src_offset;
                destination_offset = command.dst_offset;
                segment << luisa::make_unique<compute::BufferCopyCommand>(
                    source->resource.handle, destination->resource.handle, source_offset,
                    destination_offset, size);
                command_count++;
                break;
            }
            case CommandKind::TextureUpload: {
                auto *entry = _resources.find(command.resource);
                if (!is_texture_like(entry)) {
                    fail(i, command, luisa::format("'{}' is not an image or volume resource", command.resource));
                    ok = false;
                    break;
                }
                PixelStorage storage{};
                uint3 size{0u, 0u, 0u};
                luisa::string error;
                if (!resolve_texture_region(*entry, command.level, command.storage,
                                            command.size3, command.offset3, storage, size,
                                            error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                auto bytes = compute::pixel_storage_size(storage, size);
                auto &&input = command.input;
                switch (input.kind) {
                    case InputJson::Kind::None: {
                        fail(i, command, "a texture upload needs an input");
                        ok = false;
                        break;
                    }
                    case InputJson::Kind::Inline: {
                        if (input.inline_bytes.size() != bytes) {
                            fail(i, command, luisa::format("the inline payload is {} bytes, but the "
                                                           "({} {} {}) region of '{}' is {} bytes",
                                                           input.inline_bytes.size(), size.x, size.y, size.z, pixel_storage_name(storage), bytes));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::TextureUploadCommand>(
                            entry->resource.handle, storage, command.level, size,
                            input.inline_bytes.data(), command.offset3);
                        command_count++;
                        break;
                    }
                    case InputJson::Kind::Resource: {
                        auto *source = require_buffer(_resources, input.resource, i, command,
                                                      _diagnostics);
                        if (source == nullptr) {
                            ok = false;
                            break;
                        }
                        size_t source_offset = 0u;
                        size_t source_size = 0u;
                        if (!resolve_resource_region(
                                input.offset, input.size, source->resource.byte_size, bytes,
                                source_offset, source_size,
                                luisa::format("the buffer '{}'", input.resource), error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        if (source_size < bytes) {
                            fail(i, command, luisa::format("the source region is {} bytes, but the "
                                                           "({} {} {}) region of '{}' is {} bytes",
                                                           source_size, size.x, size.y, size.z, pixel_storage_name(storage), bytes));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::BufferToTextureCopyCommand>(
                            source->resource.handle, source_offset, entry->resource.handle,
                            storage, command.level, size, command.offset3);
                        command_count++;
                        break;
                    }
                    case InputJson::Kind::File: {
                        auto path = _paths.resolve_input(input.file);
                        size_t file_offset = 0u;
                        size_t file_size = 0u;
                        if (!resolve_file_region(path, input.offset, input.size, bytes,
                                                 file_offset, file_size, error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        if (file_size != bytes) {
                            fail(i, command, luisa::format("the file region is {} bytes, but the "
                                                           "({} {} {}) region of '{}' is {} bytes",
                                                           file_size, size.x, size.y, size.z, pixel_storage_name(storage), bytes));
                            ok = false;
                            break;
                        }
                        DStorageCompression compression{};
                        auto compression_name = effective_compression(input, document.config);
                        auto compression_supported = parse_dstorage_compression(
                            compression_name, compression);
                        auto full = mip_extent(entry->resource.extent, command.level);
                        // A DirectStorage texture read always starts at pixel
                        // (0, 0, 0), so only a whole mip level can take that path.
                        auto whole_level = command.offset3.x == 0u && command.offset3.y == 0u &&
                                           command.offset3.z == 0u && size.x == full.x &&
                                           size.y == full.y && size.z == full.z;
                        if (whole_level &&
                            choose_dstorage(document.config.dstorage.enabled, dstorage,
                                            _upload_stream, compression_supported,
                                            compression_name, path, file_offset, file_size,
                                            _device.backend_name(), document.config.strict,
                                            _diagnostics)) {
                            auto file = dstorage->open_file(luisa::to_string(path));
                            if (!file) {
                                fail(i, command, luisa::format("the DirectStorage extension cannot open '{}'", luisa::to_string(path)));
                                ok = false;
                                break;
                            }
                            auto read = build_dstorage_read(file.view(file_offset, file_size),
                                                            *entry, 0u, command.level, size,
                                                            compression);
                            if (!read) {
                                fail(i, command, "internal error: this resource cannot be a "
                                                 "DirectStorage destination");
                                ok = false;
                                break;
                            }
                            submit_upload(std::move(read));
                            command_count++;
                            break;
                        }
                        auto *staging = arena.allocate(bytes);
                        if (!read_file_region(path, file_offset, file_size, staging, error)) {
                            fail(i, command, std::move(error));
                            ok = false;
                            break;
                        }
                        segment << luisa::make_unique<compute::TextureUploadCommand>(
                            entry->resource.handle, storage, command.level, size, staging,
                            command.offset3);
                        command_count++;
                        break;
                    }
                }
                break;
            }
            case CommandKind::TextureDownload: {
                auto *entry = _resources.find(command.resource);
                if (!is_texture_like(entry)) {
                    fail(i, command, luisa::format("'{}' is not an image or volume resource", command.resource));
                    ok = false;
                    break;
                }
                PixelStorage storage{};
                uint3 size{0u, 0u, 0u};
                luisa::string error;
                if (!resolve_texture_region(*entry, command.level, command.storage,
                                            command.size3, command.offset3, storage, size,
                                            error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                if (enum_name_is(command.output.format, "png")) {
                    auto full = mip_extent(entry->resource.extent, command.level);
                    if (size.z > 1u || size.x != full.x || size.y != full.y ||
                        command.offset3.x != 0u || command.offset3.y != 0u) {
                        fail(i, command, "a PNG sink needs the whole 2-D mip level, written "
                                         "from pixel (0, 0)");
                        ok = false;
                        break;
                    }
                }
                auto bytes = compute::pixel_storage_size(storage, size);
                auto *target = arena.allocate(bytes);
                segment << luisa::make_unique<compute::TextureDownloadCommand>(
                    entry->resource.handle, storage, command.level, size, target,
                    command.offset3);
                command_count++;
                PendingDownload pending;
                pending.index = i;
                pending.data = target;
                pending.size = bytes;
                pending.extent = size;
                pending.storage = storage;
                pending.output = &command.output;
                pending.verify = &command.verify;
                if (!enum_name_is(command.verify.kind, "none") &&
                    !command.verify.kind.empty()) {
                    fail(i, command, "only a buffer download can be verified");
                    ok = false;
                }
                downloads.emplace_back(std::move(pending));
                break;
            }
            case CommandKind::TextureCopy: {
                auto *source = _resources.find(command.src);
                auto *destination = _resources.find(command.dst);
                if (!is_texture_like(source) || !is_texture_like(destination)) {
                    fail(i, command, luisa::format("a texture copy needs two image or volume "
                                                   "resources, got '{}' and '{}'",
                                                   command.src, command.dst));
                    ok = false;
                    break;
                }
                PixelStorage storage{};
                uint3 size{0u, 0u, 0u};
                luisa::string error;
                if (!resolve_texture_region(*source, command.src_level, command.storage,
                                            command.size3, command.offset3, storage, size,
                                            error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                // The destination level and region are checked against the same
                // storage the source agreed with.
                uint3 ignored_size{0u, 0u, 0u};
                PixelStorage ignored_storage{};
                if (!resolve_texture_region(*destination, command.dst_level, command.storage,
                                            size, command.offset3, ignored_storage,
                                            ignored_size, error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                // The wire format carries a single 3-D offset, used for both
                // sides of the copy.
                segment << luisa::make_unique<compute::TextureCopyCommand>(
                    storage, source->resource.handle, destination->resource.handle,
                    command.src_level, command.dst_level, size, command.offset3,
                    command.offset3);
                command_count++;
                break;
            }
            case CommandKind::BufferToTextureCopy:
            case CommandKind::TextureToBufferCopy: {
                auto to_texture = command.kind == CommandKind::BufferToTextureCopy;
                auto *texture = _resources.find(command.texture);
                auto *buffer = require_buffer(_resources, command.buffer, i, command,
                                              _diagnostics);
                if (buffer == nullptr || !is_texture_like(texture)) {
                    fail(i, command, luisa::format("this copy needs a buffer and an image or "
                                                   "volume resource"));
                    ok = false;
                    break;
                }
                PixelStorage storage{};
                uint3 size{0u, 0u, 0u};
                luisa::string error;
                if (!resolve_texture_region(*texture, command.level, command.storage,
                                            command.size3, command.offset3, storage, size,
                                            error)) {
                    fail(i, command, std::move(error));
                    ok = false;
                    break;
                }
                auto bytes = compute::pixel_storage_size(storage, size);
                if (command.buffer_offset > buffer->resource.byte_size ||
                    bytes > buffer->resource.byte_size - command.buffer_offset) {
                    fail(i, command, luisa::format("the {} byte region of '{}' does not fit after its "
                                                   "offset {} in the {} bytes of the buffer",
                                                   bytes, command.buffer, command.buffer_offset, buffer->resource.byte_size));
                    ok = false;
                    break;
                }
                if (to_texture) {
                    segment << luisa::make_unique<compute::BufferToTextureCopyCommand>(
                        buffer->resource.handle, command.buffer_offset,
                        texture->resource.handle, storage, command.level, size,
                        command.offset3);
                } else {
                    segment << luisa::make_unique<compute::TextureToBufferCopyCommand>(
                        buffer->resource.handle, command.buffer_offset,
                        texture->resource.handle, storage, command.level, size,
                        command.offset3);
                }
                command_count++;
                break;
            }
            case CommandKind::NativeDispatch: {
                if (!append_native_dispatch(command, i, _resources, _shaders, segment,
                                            command_count, _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::ShaderDispatch: {
                if (!append_shader_dispatch(command, i, _resources, _shaders, segment,
                                            command_count, _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::BindlessArrayUpdate: {
                if (!append_bindless_array_update(command, i, _resources, segment,
                                                  command_count, _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::MeshBuild: {
                if (!append_mesh_build(command, i, _resources, segment, command_count,
                                       _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::ProceduralPrimitiveBuild: {
                if (!append_procedural_primitive_build(command, i, _resources, segment,
                                                       command_count, _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::AccelBuild: {
                if (!append_accel_build(command, i, _resources, segment, command_count,
                                        _diagnostics)) {
                    ok = false;
                }
                break;
            }
            case CommandKind::CustomCommand: {
                if (command.uuid == luisa::to_underlying(
                                        compute::CustomCommandUUID::NATIVE_SHADER_DISPATCH)) {
                    if (!append_native_dispatch(command, i, _resources, _shaders, segment,
                                                command_count, _diagnostics)) {
                        ok = false;
                    }
                    break;
                }
                if (command.uuid == luisa::to_underlying(
                                        compute::CustomCommandUUID::DSTORAGE_READ)) {
                    auto *destination = require_buffer(_resources, command.resource, i,
                                                       command, _diagnostics);
                    if (destination == nullptr) {
                        ok = false;
                        break;
                    }
                    auto &&input = command.input;
                    if (input.kind != InputJson::Kind::File || input.file.empty()) {
                        fail(i, command, "a dstorage_read custom command needs a file input");
                        ok = false;
                        break;
                    }
                    if (command.offset > destination->resource.byte_size) {
                        fail(i, command, luisa::format("the destination offset {} exceeds the {} bytes "
                                                       "of '{}'",
                                                       command.offset, destination->resource.byte_size, command.resource));
                        ok = false;
                        break;
                    }
                    auto path = _paths.resolve_input(input.file);
                    size_t file_offset = 0u;
                    size_t size = 0u;
                    luisa::string error;
                    if (!resolve_file_region(path, input.offset, input.size,
                                             destination->resource.byte_size - command.offset,
                                             file_offset, size, error)) {
                        fail(i, command, std::move(error));
                        ok = false;
                        break;
                    }
                    DStorageCompression compression{};
                    auto compression_name = effective_compression(input, document.config);
                    auto compression_supported = parse_dstorage_compression(compression_name,
                                                                            compression);
                    if (choose_dstorage(document.config.dstorage.enabled, dstorage,
                                        _upload_stream, compression_supported,
                                        compression_name, path, file_offset, size,
                                        _device.backend_name(), document.config.strict,
                                        _diagnostics)) {
                        auto file = dstorage->open_file(luisa::to_string(path));
                        if (!file) {
                            fail(i, command, luisa::format("the DirectStorage extension cannot open '{}'", luisa::to_string(path)));
                            ok = false;
                            break;
                        }
                        auto read = build_dstorage_read(file.view(file_offset, size),
                                                        *destination, command.offset, 0u,
                                                        destination->resource.extent,
                                                        compression);
                        if (!read) {
                            fail(i, command, "internal error: this resource cannot be a "
                                             "DirectStorage destination");
                            ok = false;
                            break;
                        }
                        submit_upload(std::move(read));
                        command_count++;
                        break;
                    }
                    auto *staging = arena.allocate(size);
                    if (!read_file_region(path, file_offset, size, staging, error)) {
                        fail(i, command, std::move(error));
                        ok = false;
                        break;
                    }
                    segment << luisa::make_unique<compute::BufferUploadCommand>(
                        destination->resource.handle, command.offset, size, staging);
                    command_count++;
                    break;
                }
                _diagnostics.error(luisa::format(
                    "workflow[{}] ({}): unknown custom command uuid 0x{:08X}; registered uuids "
                    "are NATIVE_SHADER_DISPATCH (0x0600) and DSTORAGE_READ (0x0200)",
                    i, command_kind_name(command.kind), command.uuid));
                ok = false;
                break;
            }
        }
        _command_counts[luisa::to_underlying(command.kind)]++;
    }

    // Submit what is left, wait once, then write and verify the downloads: the
    // arena that owns their bytes is alive until this function returns.
    flush();
    _stream.synchronize();
    for (auto &&pending : downloads) {
        auto &&output = *pending.output;
        if (!output.discard) {
            auto path = _paths.resolve_output(output.file);
            luisa::string error;
            if (write_output(path, luisa::span{pending.data, pending.size},
                             output.format, pending.extent, pending.storage,
                             output.overwrite, error)) {
                LUISA_INFO("[workflow {}] wrote {} bytes to '{}'", pending.index,
                           pending.size, luisa::to_string(path));
            } else {
                fail(pending.index, document.workflow[pending.index], std::move(error));
                ok = false;
            }
        }
        auto &&verify = *pending.verify;
        if (pending.verify_source == nullptr || verify.kind.empty() ||
            enum_name_is(verify.kind, "none")) {
            continue;
        }
        if (enum_name_is(verify.kind, "copy")) {
            for (auto offset = size_t{0u}; offset < pending.size; offset++) {
                if (pending.data[offset] != pending.verify_source[offset]) {
                    fail(pending.index, document.workflow[pending.index],
                         luisa::format("verification failed: byte {} differs from '{}' "
                                       "(0x{:02X} != 0x{:02X})",
                                       offset, verify.source,
                                       static_cast<unsigned>(pending.data[offset]),
                                       static_cast<unsigned>(pending.verify_source[offset])));
                    ok = false;
                    break;
                }
            }
            continue;
        }
        // "linear": `dst[i] == src[i] * k + c`, both read as little-endian
        // floats, `tolerance` being an absolute epsilon.
        if (pending.size % sizeof(float) != 0u) {
            fail(pending.index, document.workflow[pending.index],
                 luisa::format("the {} downloaded bytes are not a whole number of floats, "
                               "so they cannot be verified against '{}'",
                               pending.size, verify.source));
            ok = false;
            continue;
        }
        auto count = pending.size / sizeof(float);
        auto *destination = reinterpret_cast<const float *>(pending.data);
        auto *reference = reinterpret_cast<const float *>(pending.verify_source);
        auto scale = static_cast<double>(verify.k);
        auto bias = static_cast<double>(verify.c);
        auto tolerance = static_cast<double>(verify.tolerance);
        for (auto element = size_t{0u}; element < count; element++) {
            auto expected = static_cast<double>(reference[element]) * scale + bias;
            auto actual = static_cast<double>(destination[element]);
            auto difference = actual - expected;
            if (difference > tolerance || difference < -tolerance) {
                fail(pending.index, document.workflow[pending.index],
                     luisa::format("verification failed: element {} of '{}' is {} but "
                                   "{} * {} + {} is {} (tolerance {})",
                                   element, document.workflow[pending.index].resource,
                                   actual, static_cast<double>(reference[element]),
                                   verify.k, verify.c, expected, verify.tolerance));
                ok = false;
                break;
            }
        }
    }
    _last_command_count = command_count;
    return ok && _diagnostics.ok();
}

}// namespace luisa::native_shader
