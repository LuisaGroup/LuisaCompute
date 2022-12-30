#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


typedef enum LCAccelBuildRequest {
    LC_ACCEL_BUILD_REQUEST_PREFER_UPDATE,
    LC_ACCEL_BUILD_REQUEST_FORCE_BUILD,
} LCAccelBuildRequest;

typedef enum LCAccelUsageHint {
    LC_ACCEL_USAGE_HINT_FAST_TRACE,
    LC_ACCEL_USAGE_HINT_FAST_BUILD,
} LCAccelUsageHint;

typedef enum LCMeshType {
    LC_MESH_TYPE_MESH,
    LC_MESH_TYPE_PROCEDURAL_PRIMITIVE,
} LCMeshType;

typedef enum LCPixelFormat {
    LC_PIXEL_FORMAT_R8_SINT,
    LC_PIXEL_FORMAT_R8_UINT,
    LC_PIXEL_FORMAT_R8_UNORM,
    LC_PIXEL_FORMAT_RGB8_SINT,
    LC_PIXEL_FORMAT_RGB8_UINT,
    LC_PIXEL_FORMAT_RGB8_UNORM,
    LC_PIXEL_FORMAT_RGBA8_SINT,
    LC_PIXEL_FORMAT_RGBA8_UINT,
    LC_PIXEL_FORMAT_RGBA8_UNORM,
    LC_PIXEL_FORMAT_R16_SINT,
    LC_PIXEL_FORMAT_R16_UINT,
    LC_PIXEL_FORMAT_R16_UNORM,
    LC_PIXEL_FORMAT_RG16_SINT,
    LC_PIXEL_FORMAT_RG16_UINT,
    LC_PIXEL_FORMAT_RG16_UNORM,
    LC_PIXEL_FORMAT_RGBA16_SINT,
    LC_PIXEL_FORMAT_RGBA16_UINT,
    LC_PIXEL_FORMAT_RGBA16_UNORM,
    LC_PIXEL_FORMAT_R32_SINT,
    LC_PIXEL_FORMAT_R32_UINT,
    LC_PIXEL_FORMAT_RG32_SINT,
    LC_PIXEL_FORMAT_RG32_UINT,
    LC_PIXEL_FORMAT_R16F,
    LC_PIXEL_FORMAT_RG32F,
    LC_PIXEL_FORMAT_RGBA32F,
} LCPixelFormat;

typedef enum LCPixelStorage {
    LC_PIXEL_STORAGE_BYTE1,
    LC_PIXEL_STORAGE_BYTE2,
    LC_PIXEL_STORAGE_BYTE4,
    LC_PIXEL_STORAGE_SHORT1,
    LC_PIXEL_STORAGE_SHORT2,
    LC_PIXEL_STORAGE_SHORT4,
    LC_PIXEL_STORAGE_INT1,
    LC_PIXEL_STORAGE_INT2,
    LC_PIXEL_STORAGE_INT4,
    LC_PIXEL_STORAGE_HALF1,
    LC_PIXEL_STORAGE_HALF2,
    LC_PIXEL_STORAGE_HALF4,
    LC_PIXEL_STORAGE_FLOAT1,
    LC_PIXEL_STORAGE_FLOAT2,
    LC_PIXEL_STORAGE_FLOAT4,
} LCPixelStorage;

typedef enum LCSamplerAddress {
    LC_SAMPLER_ADDRESS_EDGE,
    LC_SAMPLER_ADDRESS_REPEAT,
    LC_SAMPLER_ADDRESS_MIRROR,
    LC_SAMPLER_ADDRESS_ZERO,
} LCSamplerAddress;

typedef enum LCSamplerFilter {
    LC_SAMPLER_FILTER_POINT,
    LC_SAMPLER_FILTER_LINEAR_POINT,
    LC_SAMPLER_FILTER_LINEAR_LINEAR,
    LC_SAMPLER_FILTER_ANISOTROPIC,
} LCSamplerFilter;

typedef enum LCStreamTag {
    LC_STREAM_TAG_GRAPHICS,
    LC_STREAM_TAG_COMPUTE,
    LC_STREAM_TAG_COPY,
} LCStreamTag;

typedef struct LCCapture LCCapture;

typedef struct LCBuffer {
    uint64_t _0;
} LCBuffer;

typedef struct LCContext {
    uint64_t _0;
} LCContext;

typedef struct LCDevice {
    uint64_t _0;
} LCDevice;

typedef struct LCEvent {
    uint64_t _0;
} LCEvent;

typedef struct LCStream {
    uint64_t _0;
} LCStream;

typedef struct LCShader {
    uint64_t _0;
} LCShader;

typedef struct LCAccelBuildModificationFlags {
    uint32_t bits;
} LCAccelBuildModificationFlags;
#define LCAccelBuildModificationFlags_EMPTY (LCAccelBuildModificationFlags){ .bits = (uint32_t)0 }
#define LCAccelBuildModificationFlags_MESH (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 0) }
#define LCAccelBuildModificationFlags_TRANSFORM (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 1) }
#define LCAccelBuildModificationFlags_VISIBILITY_ON (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 2) }
#define LCAccelBuildModificationFlags_VISIBILITY_OFF (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 3) }
#define LCAccelBuildModificationFlags_VISIBILITY (LCAccelBuildModificationFlags){ .bits = (uint32_t)((LCAccelBuildModificationFlags_VISIBILITY_ON).bits | (LCAccelBuildModificationFlags_VISIBILITY_OFF).bits) }

typedef struct LCAccelBuildModification {
    uint32_t index;
    struct LCAccelBuildModificationFlags flags;
    uint64_t mesh;
    float affine[12];
} LCAccelBuildModification;

typedef struct LCSampler {
    enum LCSamplerFilter filter;
    enum LCSamplerAddress address;
} LCSampler;

typedef struct LCBufferUploadCommand {
    struct LCBuffer buffer;
    size_t offset;
    size_t size;
    const uint8_t *data;
} LCBufferUploadCommand;

typedef struct LCBufferDownloadCommand {
    struct LCBuffer buffer;
    size_t offset;
    size_t size;
    uint8_t *data;
} LCBufferDownloadCommand;

typedef struct LCBufferCopyCommand {
    struct LCBuffer src;
    size_t src_offset;
    struct LCBuffer dst;
    size_t dst_offset;
    size_t size;
} LCBufferCopyCommand;

typedef struct LCTexture {
    uint64_t _0;
} LCTexture;

typedef struct LCBufferToTextureCopyCommand {
    struct LCBuffer buffer;
    size_t buffer_offset;
    struct LCTexture texture;
    enum LCPixelStorage storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
} LCBufferToTextureCopyCommand;

typedef struct LCTextureToBufferCopyCommand {
    struct LCBuffer buffer;
    size_t buffer_offset;
    struct LCTexture texture;
    enum LCPixelStorage storage;
    uint32_t texture_level;
    uint32_t texture_size[3];
} LCTextureToBufferCopyCommand;

typedef struct LCTextureUploadCommand {
    struct LCTexture texture;
    enum LCPixelStorage storage;
    uint32_t level;
    uint32_t size[3];
    const uint8_t *data;
} LCTextureUploadCommand;

typedef struct LCTextureDownloadCommand {
    struct LCTexture texture;
    enum LCPixelStorage storage;
    uint32_t level;
    uint32_t size[3];
    uint8_t *data;
} LCTextureDownloadCommand;

typedef struct LCBufferArgument {
    struct LCBuffer buffer;
    size_t offset;
    size_t size;
} LCBufferArgument;

typedef struct LCTextureArgument {
    struct LCTexture texture;
    uint32_t level;
} LCTextureArgument;

typedef struct LCUniformArgument {
    const uint8_t *data;
    size_t size;
} LCUniformArgument;

typedef struct LCAccel {
    uint64_t _0;
} LCAccel;

typedef struct LCBindlessArray {
    uint64_t _0;
} LCBindlessArray;

typedef enum LCArgument_Tag {
    LC_ARGUMENT_BUFFER,
    LC_ARGUMENT_TEXTURE,
    LC_ARGUMENT_UNIFORM,
    LC_ARGUMENT_ACCEL,
    LC_ARGUMENT_BINDLESS_ARRAY,
} LCArgument_Tag;

typedef struct LCArgument {
    LCArgument_Tag tag;
    union {
        struct {
            struct LCBufferArgument buffer;
        };
        struct {
            struct LCTextureArgument texture;
        };
        struct {
            struct LCUniformArgument uniform;
        };
        struct {
            struct LCAccel accel;
        };
        struct {
            struct LCBindlessArray bindless_array;
        };
    };
} LCArgument;

typedef struct LCShaderDispatchCommand {
    struct LCShader shader;
    uint32_t dispatch_size[3];
    const struct LCArgument *args;
    size_t args_count;
} LCShaderDispatchCommand;

typedef struct LCMesh {
    uint64_t _0;
} LCMesh;

typedef struct LCMeshBuildCommand {
    struct LCMesh mesh;
    enum LCAccelBuildRequest request;
    struct LCBuffer vertex_buffer;
    size_t vertex_buffer_offset;
    size_t vertex_buffer_size;
    struct LCBuffer index_buffer;
    size_t index_buffer_offset;
    size_t index_buffer_size;
} LCMeshBuildCommand;

typedef struct LCAccelBuildCommand {
    struct LCAccel accel;
    enum LCAccelBuildRequest request;
    uint32_t instance_count;
    const struct LCAccelBuildModification *modifications;
    size_t modifications_count;
} LCAccelBuildCommand;

typedef enum LCCommand_Tag {
    LC_COMMAND_BUFFER_UPLOAD,
    LC_COMMAND_BUFFER_DOWNLOAD,
    LC_COMMAND_BUFFER_COPY,
    LC_COMMAND_BUFFER_TO_TEXTURE_COPY,
    LC_COMMAND_TEXTURE_TO_BUFFER_COPY,
    LC_COMMAND_TEXTURE_UPLOAD,
    LC_COMMAND_TEXTURE_DOWNLOAD,
    LC_COMMAND_SHADER_DISPATCH,
    LC_COMMAND_MESH_BUILD,
    LC_COMMAND_ACCEL_BUILD,
    LC_COMMAND_BINDLESS_ARRAY_UPDATE,
} LCCommand_Tag;

typedef struct LCCommand {
    LCCommand_Tag tag;
    union {
        struct {
            struct LCBufferUploadCommand buffer_upload;
        };
        struct {
            struct LCBufferDownloadCommand buffer_download;
        };
        struct {
            struct LCBufferCopyCommand buffer_copy;
        };
        struct {
            struct LCBufferToTextureCopyCommand buffer_to_texture_copy;
        };
        struct {
            struct LCTextureToBufferCopyCommand texture_to_buffer_copy;
        };
        struct {
            struct LCTextureUploadCommand texture_upload;
        };
        struct {
            struct LCTextureDownloadCommand texture_download;
        };
        struct {
            struct LCShaderDispatchCommand shader_dispatch;
        };
        struct {
            struct LCMeshBuildCommand mesh_build;
        };
        struct {
            struct LCAccelBuildCommand accel_build;
        };
        struct {
            struct LCBindlessArray bindless_array_update;
        };
    };
} LCCommand;

typedef struct LCCommandList {
    const struct LCCommand *commands;
    size_t commands_count;
} LCCommandList;

typedef struct LCAppContext {
    void *gc_context;
    void *ir_context;
} LCAppContext;

typedef struct LCIrModule {
    uint64_t _0;
} LCIrModule;

typedef struct LCNodeRef {
    uint64_t _0;
} LCNodeRef;

typedef struct LCKernelModule {
    struct LCIrModule ir_module;
    const struct LCCapture *captured;
    size_t captured_count;
    const struct LCNodeRef *args;
    size_t args_count;
    const struct LCNodeRef *shared;
    size_t shared_count;
} LCKernelModule;
