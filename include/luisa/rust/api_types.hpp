#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>


namespace luisa::compute::api {

static const uint64_t INVALID_RESOURCE_HANDLE = UINT64_MAX;

enum class AccelBuildRequest {
    PREFER_UPDATE,
    FORCE_BUILD,
};

enum class AccelUsageHint {
    FAST_TRACE,
    FAST_BUILD,
};

enum class BindlessArrayUpdateOperation {
    NONE,
    EMPLACE,
    REMOVE,
};

enum class PixelFormat {
    R8_SINT,
    R8_UINT,
    R8_UNORM,
    RG8_SINT,
    RG8_UINT,
    RG8_UNORM,
    RGBA8_SINT,
    RGBA8_UINT,
    RGBA8_UNORM,
    R16_SINT,
    R16_UINT,
    R16_UNORM,
    RG16_SINT,
    RG16_UINT,
    RG16_UNORM,
    RGBA16_SINT,
    RGBA16_UINT,
    RGBA16_UNORM,
    R32_SINT,
    R32_UINT,
    RG32_SINT,
    RG32_UINT,
    RGBA32_SINT,
    RGBA32_UINT,
    R16F,
    RG16F,
    RGBA16F,
    R32F,
    RG32F,
    RGBA32F,
};

enum class PixelStorage {
    BYTE1,
    BYTE2,
    BYTE4,
    SHORT1,
    SHORT2,
    SHORT4,
    INT1,
    INT2,
    INT4,
    HALF1,
    HALF2,
    HALF4,
    FLOAT1,
    FLOAT2,
    FLOAT4,
};

enum class SamplerAddress {
    EDGE,
    REPEAT,
    MIRROR,
    ZERO,
};

enum class SamplerFilter {
    POINT,
    LINEAR_POINT,
    LINEAR_LINEAR,
    ANISOTROPIC,
};

enum class StreamTag {
    GRAPHICS,
    COMPUTE,
    COPY,
};

struct Buffer {
    uint64_t _0;
};

struct Context {
    uint64_t _0;
};

struct Device {
    uint64_t _0;
};

struct Event {
    uint64_t _0;
};

struct Stream {
    uint64_t _0;
};

struct Shader {
    uint64_t _0;
};

struct Swapchain {
    uint64_t _0;
};

struct AccelBuildModificationFlags {
    uint32_t bits;

    explicit operator bool() const {
        return !!bits;
    }
    AccelBuildModificationFlags operator~() const {
        return AccelBuildModificationFlags { static_cast<decltype(bits)>(~bits) };
    }
    AccelBuildModificationFlags operator|(const AccelBuildModificationFlags& other) const {
        return AccelBuildModificationFlags { static_cast<decltype(bits)>(this->bits | other.bits) };
    }
    AccelBuildModificationFlags& operator|=(const AccelBuildModificationFlags& other) {
        *this = (*this | other);
        return *this;
    }
    AccelBuildModificationFlags operator&(const AccelBuildModificationFlags& other) const {
        return AccelBuildModificationFlags { static_cast<decltype(bits)>(this->bits & other.bits) };
    }
    AccelBuildModificationFlags& operator&=(const AccelBuildModificationFlags& other) {
        *this = (*this & other);
        return *this;
    }
    AccelBuildModificationFlags operator^(const AccelBuildModificationFlags& other) const {
        return AccelBuildModificationFlags { static_cast<decltype(bits)>(this->bits ^ other.bits) };
    }
    AccelBuildModificationFlags& operator^=(const AccelBuildModificationFlags& other) {
        *this = (*this ^ other);
        return *this;
    }
};
static const AccelBuildModificationFlags AccelBuildModificationFlags_EMPTY = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)0 };
static const AccelBuildModificationFlags AccelBuildModificationFlags_PRIMITIVE = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 0) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_TRANSFORM = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 1) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_OPAQUE_ON = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 2) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_OPAQUE_OFF = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 3) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_VISIBILITY = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 4) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_USER_ID = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 5) };

struct AccelBuildModification {
    uint32_t index;
    uint32_t user_id;
    AccelBuildModificationFlags flags;
    uint32_t visibility;
    uint64_t mesh;
    float affine[12];
};

struct Sampler {
    SamplerFilter filter;
    SamplerAddress address;
};

struct BufferUploadCommand {
    Buffer buffer;
    size_t offset;
    size_t size;
    const uint8_t *data;
};

struct BufferDownloadCommand {
    Buffer buffer;
    size_t offset;
    size_t size;
    uint8_t *data;
};

struct BufferCopyCommand {
    Buffer src;
    size_t src_offset;
    Buffer dst;
    size_t dst_offset;
    size_t size;
};

struct Texture {
    uint64_t _0;
};

struct BufferToTextureCopyCommand {
    Buffer buffer;
    size_t buffer_offset;
    Texture texture;
    PixelStorage storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
};

struct TextureToBufferCopyCommand {
    Buffer buffer;
    size_t buffer_offset;
    Texture texture;
    PixelStorage storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
};

struct TextureUploadCommand {
    Texture texture;
    PixelStorage storage;
    uint32_t level;
    uint32_t size[3];
    const uint8_t *data;
};

struct TextureDownloadCommand {
    Texture texture;
    PixelStorage storage;
    uint32_t level;
    uint32_t size[3];
    uint8_t *data;
};

struct TextureCopyCommand {
    PixelStorage storage;
    Texture src;
    Texture dst;
    uint32_t size[3];
    uint32_t src_level;
    uint32_t dst_level;
};

struct BufferArgument {
    Buffer buffer;
    size_t offset;
    size_t size;
};

struct TextureArgument {
    Texture texture;
    uint32_t level;
};

struct UniformArgument {
    const uint8_t *data;
    size_t size;
};

struct BindlessArray {
    uint64_t _0;
};

struct Accel {
    uint64_t _0;
};

struct Argument {
    enum class Tag {
        BUFFER,
        TEXTURE,
        UNIFORM,
        BINDLESS_ARRAY,
        ACCEL,
    };

    struct Buffer_Body {
        BufferArgument _0;
    };

    struct Texture_Body {
        TextureArgument _0;
    };

    struct Uniform_Body {
        UniformArgument _0;
    };

    struct BindlessArray_Body {
        BindlessArray _0;
    };

    struct Accel_Body {
        Accel _0;
    };

    Tag tag;
    union {
        Buffer_Body BUFFER;
        Texture_Body TEXTURE;
        Uniform_Body UNIFORM;
        BindlessArray_Body BINDLESS_ARRAY;
        Accel_Body ACCEL;
    };
};

struct ShaderDispatchCommand {
    Shader shader;
    uint32_t dispatch_size[3];
    const Argument *args;
    size_t args_count;
};

struct Mesh {
    uint64_t _0;
};

struct MeshBuildCommand {
    Mesh mesh;
    AccelBuildRequest request;
    Buffer vertex_buffer;
    size_t vertex_buffer_offset;
    size_t vertex_buffer_size;
    size_t vertex_stride;
    Buffer index_buffer;
    size_t index_buffer_offset;
    size_t index_buffer_size;
    size_t index_stride;
};

struct ProceduralPrimitive {
    uint64_t _0;
};

struct ProceduralPrimitiveBuildCommand {
    ProceduralPrimitive handle;
    AccelBuildRequest request;
    Buffer aabb_buffer;
    size_t aabb_buffer_offset;
    size_t aabb_count;
};

struct AccelBuildCommand {
    Accel accel;
    AccelBuildRequest request;
    uint32_t instance_count;
    const AccelBuildModification *modifications;
    size_t modifications_count;
    bool update_instance_buffer_only;
};

struct BindlessArrayUpdateBuffer {
    BindlessArrayUpdateOperation op;
    Buffer handle;
    size_t offset;
};

struct BindlessArrayUpdateTexture {
    BindlessArrayUpdateOperation op;
    Texture handle;
    Sampler sampler;
};

struct BindlessArrayUpdateModification {
    size_t slot;
    BindlessArrayUpdateBuffer buffer;
    BindlessArrayUpdateTexture tex2d;
    BindlessArrayUpdateTexture tex3d;
};

struct BindlessArrayUpdateCommand {
    BindlessArray handle;
    const BindlessArrayUpdateModification *modifications;
    size_t modifications_count;
};

struct Command {
    enum class Tag {
        BUFFER_UPLOAD,
        BUFFER_DOWNLOAD,
        BUFFER_COPY,
        BUFFER_TO_TEXTURE_COPY,
        TEXTURE_TO_BUFFER_COPY,
        TEXTURE_UPLOAD,
        TEXTURE_DOWNLOAD,
        TEXTURE_COPY,
        SHADER_DISPATCH,
        MESH_BUILD,
        PROCEDURAL_PRIMITIVE_BUILD,
        ACCEL_BUILD,
        BINDLESS_ARRAY_UPDATE,
    };

    struct BufferUpload_Body {
        BufferUploadCommand _0;
    };

    struct BufferDownload_Body {
        BufferDownloadCommand _0;
    };

    struct BufferCopy_Body {
        BufferCopyCommand _0;
    };

    struct BufferToTextureCopy_Body {
        BufferToTextureCopyCommand _0;
    };

    struct TextureToBufferCopy_Body {
        TextureToBufferCopyCommand _0;
    };

    struct TextureUpload_Body {
        TextureUploadCommand _0;
    };

    struct TextureDownload_Body {
        TextureDownloadCommand _0;
    };

    struct TextureCopy_Body {
        TextureCopyCommand _0;
    };

    struct ShaderDispatch_Body {
        ShaderDispatchCommand _0;
    };

    struct MeshBuild_Body {
        MeshBuildCommand _0;
    };

    struct ProceduralPrimitiveBuild_Body {
        ProceduralPrimitiveBuildCommand _0;
    };

    struct AccelBuild_Body {
        AccelBuildCommand _0;
    };

    struct BindlessArrayUpdate_Body {
        BindlessArrayUpdateCommand _0;
    };

    Tag tag;
    union {
        BufferUpload_Body BUFFER_UPLOAD;
        BufferDownload_Body BUFFER_DOWNLOAD;
        BufferCopy_Body BUFFER_COPY;
        BufferToTextureCopy_Body BUFFER_TO_TEXTURE_COPY;
        TextureToBufferCopy_Body TEXTURE_TO_BUFFER_COPY;
        TextureUpload_Body TEXTURE_UPLOAD;
        TextureDownload_Body TEXTURE_DOWNLOAD;
        TextureCopy_Body TEXTURE_COPY;
        ShaderDispatch_Body SHADER_DISPATCH;
        MeshBuild_Body MESH_BUILD;
        ProceduralPrimitiveBuild_Body PROCEDURAL_PRIMITIVE_BUILD;
        AccelBuild_Body ACCEL_BUILD;
        BindlessArrayUpdate_Body BINDLESS_ARRAY_UPDATE;
    };
};

struct CommandList {
    const Command *commands;
    size_t commands_count;
};

struct KernelModule {
    uint64_t ptr;
};

struct CreatedResourceInfo {
    uint64_t handle;
    void *native_handle;
};

struct CreatedBufferInfo {
    CreatedResourceInfo resource;
    size_t element_stride;
    size_t total_size_bytes;
};

struct CreatedShaderInfo {
    CreatedResourceInfo resource;
    uint32_t block_size[3];
};

struct CreatedSwapchainInfo {
    CreatedResourceInfo resource;
    PixelStorage storage;
};

struct AccelOption {
    AccelUsageHint hint;
    bool allow_compaction;
    bool allow_update;
};

struct ShaderOption {
    bool enable_cache;
    bool enable_fast_math;
    bool enable_debug_info;
    bool compile_only;
    const char *name;
};

using DispatchCallback = void(*)(uint8_t*);

struct DeviceInterface {
    Device device;
    void (*destroy_device)(DeviceInterface);
    CreatedBufferInfo (*create_buffer)(Device, const void*, size_t);
    void (*destroy_buffer)(Device, Buffer);
    CreatedResourceInfo (*create_texture)(Device,
                                          PixelFormat,
                                          uint32_t,
                                          uint32_t,
                                          uint32_t,
                                          uint32_t,
                                          uint32_t,
                                          bool);
    void *(*native_handle)(Device);
    uint32_t (*compute_warp_size)(Device);
    void (*destroy_texture)(Device, Texture);
    CreatedResourceInfo (*create_bindless_array)(Device, size_t);
    void (*destroy_bindless_array)(Device, BindlessArray);
    CreatedResourceInfo (*create_stream)(Device, StreamTag);
    void (*destroy_stream)(Device, Stream);
    void (*synchronize_stream)(Device, Stream);
    void (*dispatch)(Device, Stream, CommandList, DispatchCallback, uint8_t*);
    CreatedSwapchainInfo (*create_swapchain)(Device,
                                             uint64_t,
                                             Stream,
                                             uint32_t,
                                             uint32_t,
                                             bool,
                                             bool,
                                             uint32_t);
    void (*present_display_in_stream)(Device, Stream, Swapchain, Texture);
    void (*destroy_swapchain)(Device, Swapchain);
    CreatedShaderInfo (*create_shader)(Device, KernelModule, const ShaderOption*);
    void (*destroy_shader)(Device, Shader);
    CreatedResourceInfo (*create_event)(Device);
    void (*destroy_event)(Device, Event);
    void (*signal_event)(Device, Event, Stream, uint64_t);
    void (*synchronize_event)(Device, Event, uint64_t);
    void (*wait_event)(Device, Event, Stream, uint64_t);
    bool (*is_event_completed)(Device, Event, uint64_t);
    CreatedResourceInfo (*create_mesh)(Device, const AccelOption*);
    void (*destroy_mesh)(Device, Mesh);
    CreatedResourceInfo (*create_procedural_primitive)(Device, const AccelOption*);
    void (*destroy_procedural_primitive)(Device, ProceduralPrimitive);
    CreatedResourceInfo (*create_accel)(Device, const AccelOption*);
    void (*destroy_accel)(Device, Accel);
    char *(*query)(Device, const char*);
};

struct LoggerMessage {
    const char *target;
    const char *level;
    const char *message;
};

struct LibInterface {
    void *inner;
    void (*set_logger_callback)(void(*)(LoggerMessage));
    Context (*create_context)(const char*);
    void (*destroy_context)(Context);
    DeviceInterface (*create_device)(Context, const char*, const char*);
    void (*free_string)(char*);
};

} // namespace luisa::compute::api
