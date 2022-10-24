#pragma once
#include <core/platform.h>
#include <stdint.h>
#include <stddef.h>

// Uppercase names prefixed with underscores are reserved for the standard library.
#define LUISA_API_DECL_TYPE(TypeName) \
    typedef struct TypeName##_st {    \
        uint64_t __dummy;             \
    } TypeName##_st;                  \
    typedef TypeName##_st *TypeName

LUISA_API_DECL_TYPE(LCType);
LUISA_API_DECL_TYPE(LCExpression);
LUISA_API_DECL_TYPE(LCConstantData);
LUISA_API_DECL_TYPE(LCStmt);

LUISA_API_DECL_TYPE(LCContext);
LUISA_API_DECL_TYPE(LCDevice);
LUISA_API_DECL_TYPE(LCShader);
LUISA_API_DECL_TYPE(LCBuffer);
LUISA_API_DECL_TYPE(LCTexture);
LUISA_API_DECL_TYPE(LCStream);
LUISA_API_DECL_TYPE(LCEvent);
LUISA_API_DECL_TYPE(LCBindlessArray);
LUISA_API_DECL_TYPE(LCMesh);
LUISA_API_DECL_TYPE(LCAccel);
LUISA_API_DECL_TYPE(LCIRModule);
typedef size_t LCNodeRef;

#undef LUISA_API_DECL_TYPE

typedef enum LCAccelUsageHint {
    LC_FAST_TRACE, // build with best quality
    LC_FAST_UPDATE,// optimize for frequent update, usually with compaction
    LC_FAST_BUILD  // optimize for frequent rebuild, maybe without compaction
} LCAccelUsageHint;
typedef enum LCAccelBuildRequest {
    LC_PREFER_UPDATE,
    LC_FORCE_BUILD,
} LCAccelBuildRequest;

typedef enum LCPixelStorage {

    LC_BYTE1,
    LC_BYTE2,
    LC_BYTE4,

    LC_SHORT1,
    LC_SHORT2,
    LC_SHORT4,

    LC_INT1,
    LC_INT2,
    LC_INT4,

    LC_HALF1,
    LC_HALF2,
    LC_HALF4,

    LC_FLOAT1,
    LC_FLOAT2,
    LC_FLOAT4
} LCPixelStorage;

typedef enum LCPixelFormat {

    LC_R8SInt,
    LC_R8UInt,
    LC_R8UNorm,

    LC_RG8SInt,
    LC_RG8UInt,
    LC_RG8UNorm,

    LC_RGBA8SInt,
    LC_RGBA8UInt,
    LC_RGBA8UNorm,

    LC_R16SInt,
    LC_R16UInt,
    LC_R16UNorm,

    LC_RG16SInt,
    LC_RG16UInt,
    LC_RG16UNorm,

    LC_RGBA16SInt,
    LC_RGBA16UInt,
    LC_RGBA16UNorm,

    LC_R32SInt,
    LC_R32UInt,

    LC_RG32SInt,
    LC_RG32UInt,

    LC_RGBA32SInt,
    LC_RGBA32UInt,

    LC_R16F,
    LC_RG16F,
    LC_RGBA16F,

    LC_R32F,
    LC_RG32F,
    LC_RGBA32F
} LCPixelFormat;

typedef struct lc_uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} lc_uint3;

typedef enum LCAccelBuildModficationFlags {
    LC_ACCEL_MESH = 1u << 0u,
    LC_ACCEL_TRANSFORM = 1u << 1u,
    LC_ACCEL_VISIBILITY_ON = 1u << 2u,
    LC_ACCEL_VISIBILITY_OFF = 1u << 3u,
    LC_ACCEL_VISIBILITY = LC_ACCEL_VISIBILITY_ON | LC_ACCEL_VISIBILITY_OFF
} LCAccelBuildModficationFlags;

typedef struct LCAccelBuildModification {
    uint32_t index;
    LCAccelBuildModficationFlags flags;
    uint64_t mesh;
    float affine[12];
} LCAccelBuildModification;

typedef enum LCSamplerFilter {
    LC_POINT,
    LC_LINEAR_POINT,
    LC_LINEAR_LINEAR,
    LC_ANISOTROPIC
} LCSamplerFilter;

typedef enum LCSamplerAddress {
    LC_EDGE,
    LC_REPEAT,
    LC_MIRROR,
    LC_ZERO
} LCSamplerAddress;

typedef struct LCSampler {
    LCSamplerFilter filter;
    LCSamplerAddress address;
} LCSampler;

typedef enum LCArgumentTag {
    LC_BUFFER,
    LC_TEXTURE,
    LC_UNIFORM,
    LC_ACCEL,
    LC_BINDLESS_ARRAY,
} LCArgumentTag;

typedef struct LCBufferArgument {
    LCBuffer buffer;
    size_t offset;
    size_t size;
} LCBufferArgument;

typedef struct LCTextureArgument {
    LCTexture texture;
    uint32_t level;
} LCTextureArgument;

typedef struct LCUniformArgument {
    const uint8_t *data;
    size_t size;
} LCUniformArgument;

typedef struct LCBindlessArrayArgument {
    LCBindlessArray array;
} LCBindlessArrayArgument;

typedef struct LCAccelArgument {
    LCAccel accel;
} LCAccelArgument;
typedef struct LCArgument {
    LCArgumentTag tag;
    union {
        LCBufferArgument buffer;
        LCTextureArgument texture;
        LCUniformArgument uniform;
        LCAccelArgument accel;
        LCBindlessArrayArgument bindless_array;
    };
} LCArgument;
typedef struct LCCapture {
    LCArgumentTag tag;
    LCNodeRef node;
    union {
        LCBufferArgument buffer;
        LCTextureArgument texture;
        LCUniformArgument uniform;
        LCAccelArgument accel;
        LCBindlessArrayArgument bindless_array;
    };
} LCCapture;

typedef struct LCKernelModule {
    LCIRModule m;
    const LCCapture *captured;
    size_t captured_count;
    const LCNodeRef *args;
    size_t arg_count;
    const LCNodeRef *shared;
    size_t shared_count;
} LCKernelModule;

typedef struct LCCallableModule {
    LCIRModule m;
    const LCNodeRef *args;
    size_t arg_count;
} LCCallableModule;

typedef struct LCBufferUploadCommand {
    LCBuffer buffer;
    size_t offset;
    size_t size;
    const void *data;
} LCBufferUploadCommand;

typedef struct LCBufferDownloadCommand {
    LCBuffer buffer;
    size_t offset;
    size_t size;
    void *data;
} LCBufferDownloadCommand;
typedef struct LCBufferCopyCommand {
    LCBuffer src;
    size_t src_offset;
    LCBuffer dst;
    size_t dst_offset;
    size_t size;
} LCBufferCopyCommand;
typedef struct LCBufferToTextureCopyCommand {
    LCBuffer src;
    size_t src_offset;
    LCTexture dst;
    LCPixelStorage pixel_storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
} LCBufferToTextureCopyCommand;

typedef struct LCTextureToBufferCopyCommand {
    LCBuffer dst;
    size_t dst_offset;
    LCTexture src;
    LCPixelStorage pixel_storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
} LCTextureToBufferCopyCommand;

typedef struct LCTextureCopyCommand {
    LCPixelStorage _storage;
    LCTexture src;
    LCTexture dst;
    uint32_t size[3];
    uint32_t src_level;
    uint32_t dst_level;
} LCTextureCopyCommand;

typedef struct LCTextureUploadCommand {
    LCTexture texture;
    LCPixelStorage _storage;
    uint32_t level;
    uint32_t size[3];
    const void *data;
} LCTextureUploadCommand;

typedef struct LCTextureDownloadCommand {
    LCTexture texture;
    LCPixelStorage _storage;
    uint32_t level;
    uint32_t size[3];
    const void *data;
} LCTextureDownloadCommand;

typedef struct LCShaderDispatchCommand {
    LCShader shader;
    LCKernelModule m;
    uint32_t dispatch_size[3];
    LCArgument *args;
    size_t arg_count;
} LCShaderDispatchCommand;

typedef struct LCMeshBuildCommand {
    LCMesh mesh;
    LCAccelBuildRequest request;
    LCBuffer vertex_buffer;
    size_t vertex_buffer_offset;
    size_t vertex_buffer_size;
    LCBuffer triangle_buffer;
    size_t triangle_buffer_offset;
    size_t triangle_buffer_size;
} LCMeshBuildCommand;

typedef struct LCAccelBuildCommand {
    LCAccel accel;
    uint32_t instance_count;
    LCAccelBuildRequest request;
    LCAccelBuildModification *modifications;
    size_t modification_count;
} LCAccelBuildCommand;

typedef enum LCCommandTag {
    LC_BUFFER_UPLOAD,
    LC_BUFFER_DOWNLOAD,
    LC_BUFFER_COPY,
    LC_BUFFER_TO_TEXTURE_COPY,
    LC_TEXTURE_TO_BUFFER_COPY,
    LC_TEXTURE_COPY,
    LC_TEXTURE_UPLOAD,
    LC_TEXTURE_DOWNLOAD,
    LC_SHADER_DISPATCH,
    LC_MESH_BUILD,
    LC_ACCEL_BUILD,
} LCCommandTag;
typedef struct LCCommand {
    LCCommandTag tag;
    union {
        LCBufferUploadCommand buffer_upload;
        LCBufferDownloadCommand buffer_download;
        LCBufferCopyCommand buffer_copy;
        LCBufferToTextureCopyCommand buffer_to_texture_copy;
        LCTextureToBufferCopyCommand texture_to_buffer_copy;
        LCTextureCopyCommand texture_copy;
        LCTextureUploadCommand texture_upload;
        LCTextureDownloadCommand texture_download;
        LCShaderDispatchCommand shader_dispatch;
        LCMeshBuildCommand mesh_build;
        LCAccelBuildCommand accel_build;
    };
} LCCommand;

typedef struct LCCommandList {
    LCCommand *commands;
    size_t command_count;
} LCCommandList;