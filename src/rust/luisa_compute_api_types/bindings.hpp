#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>


namespace luisa::compute::api {

enum class AccelBuildRequest {
    PREFER_UPDATE,
    FORCE_BUILD,
};

enum class AccelUsageHint {
    FAST_TRACE,
    FAST_BUILD,
    FAST_UPDATE,
};

enum class PixelFormat {
    R8_SINT,
    R8_UINT,
    R8_UNORM,
    RGB8_SINT,
    RGB8_UINT,
    RGB8_UNORM,
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
    R16F,
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
static const AccelBuildModificationFlags AccelBuildModificationFlags_MESH = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 0) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_TRANSFORM = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 1) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_VISIBILITY_ON = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 2) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_VISIBILITY_OFF = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 3) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_OPAQUE_ON = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 4) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_OPAQUE_OFF = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)(1 << 5) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_VISIBILITY = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)((AccelBuildModificationFlags_VISIBILITY_ON).bits | (AccelBuildModificationFlags_VISIBILITY_OFF).bits) };
static const AccelBuildModificationFlags AccelBuildModificationFlags_OPAQUE = AccelBuildModificationFlags{ /* .bits = */ (uint32_t)((AccelBuildModificationFlags_OPAQUE_ON).bits | (AccelBuildModificationFlags_OPAQUE_OFF).bits) };

struct AccelBuildModification {
    uint32_t index;
    AccelBuildModificationFlags flags;
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

struct Accel {
    uint64_t _0;
};

struct BindlessArray {
    uint64_t _0;
};

struct Argument {
    enum class Tag {
        BUFFER,
        TEXTURE,
        UNIFORM,
        ACCEL,
        BINDLESS_ARRAY,
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

    struct Accel_Body {
        Accel _0;
    };

    struct BindlessArray_Body {
        BindlessArray _0;
    };

    Tag tag;
    union {
        Buffer_Body BUFFER;
        Texture_Body TEXTURE;
        Uniform_Body UNIFORM;
        Accel_Body ACCEL;
        BindlessArray_Body BINDLESS_ARRAY;
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

struct AccelBuildCommand {
    Accel accel;
    AccelBuildRequest request;
    uint32_t instance_count;
    const AccelBuildModification *modifications;
    size_t modifications_count;
    bool build_accel;
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
        SHADER_DISPATCH,
        MESH_BUILD,
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

    struct ShaderDispatch_Body {
        ShaderDispatchCommand _0;
    };

    struct MeshBuild_Body {
        MeshBuildCommand _0;
    };

    struct AccelBuild_Body {
        AccelBuildCommand _0;
    };

    struct BindlessArrayUpdate_Body {
        BindlessArray _0;
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
        ShaderDispatch_Body SHADER_DISPATCH;
        MeshBuild_Body MESH_BUILD;
        AccelBuild_Body ACCEL_BUILD;
        BindlessArrayUpdate_Body BINDLESS_ARRAY_UPDATE;
    };
};

struct CommandList {
    const Command *commands;
    size_t commands_count;
};

struct AppContext {
    void *gc_context;
    void *ir_context;
};

struct KernelModule {
    uint64_t ptr;
};

} // namespace luisa::compute::api
