#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


#define LCINVALID_RESOURCE_HANDLE UINT64_MAX

typedef enum LCAccelBuildRequest {
    LC_ACCEL_BUILD_REQUEST_PREFER_UPDATE,
    LC_ACCEL_BUILD_REQUEST_FORCE_BUILD,
} LCAccelBuildRequest;

typedef enum LCAccelUsageHint {
    LC_ACCEL_USAGE_HINT_FAST_TRACE,
    LC_ACCEL_USAGE_HINT_FAST_BUILD,
} LCAccelUsageHint;

typedef enum LCBindlessArrayUpdateOperation {
    LC_BINDLESS_ARRAY_UPDATE_OPERATION_NONE,
    LC_BINDLESS_ARRAY_UPDATE_OPERATION_EMPLACE,
    LC_BINDLESS_ARRAY_UPDATE_OPERATION_REMOVE,
} LCBindlessArrayUpdateOperation;

typedef enum LCPixelFormat {
    LC_PIXEL_FORMAT_R8_SINT,
    LC_PIXEL_FORMAT_R8_UINT,
    LC_PIXEL_FORMAT_R8_UNORM,
    LC_PIXEL_FORMAT_RG8_SINT,
    LC_PIXEL_FORMAT_RG8_UINT,
    LC_PIXEL_FORMAT_RG8_UNORM,
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
    LC_PIXEL_FORMAT_RGBA32_SINT,
    LC_PIXEL_FORMAT_RGBA32_UINT,
    LC_PIXEL_FORMAT_R16F,
    LC_PIXEL_FORMAT_RG16F,
    LC_PIXEL_FORMAT_RGBA16F,
    LC_PIXEL_FORMAT_R32F,
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

typedef struct LCSwapchain {
    uint64_t _0;
} LCSwapchain;

typedef struct LCAccelBuildModificationFlags {
    uint32_t bits;
} LCAccelBuildModificationFlags;
#define LCAccelBuildModificationFlags_EMPTY (LCAccelBuildModificationFlags){ .bits = (uint32_t)0 }
#define LCAccelBuildModificationFlags_PRIMITIVE (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 0) }
#define LCAccelBuildModificationFlags_TRANSFORM (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 1) }
#define LCAccelBuildModificationFlags_OPAQUE_ON (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 2) }
#define LCAccelBuildModificationFlags_OPAQUE_OFF (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 3) }
#define LCAccelBuildModificationFlags_VISIBILITY (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 4) }
#define LCAccelBuildModificationFlags_USER_ID (LCAccelBuildModificationFlags){ .bits = (uint32_t)(1 << 5) }

typedef struct LCAccelBuildModification {
    uint32_t index;
    uint32_t user_id;
    struct LCAccelBuildModificationFlags flags;
    uint32_t visibility;
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

typedef struct LCTextureCopyCommand {
    enum LCPixelStorage storage;
    struct LCTexture src;
    struct LCTexture dst;
    uint32_t size[3];
    uint32_t src_level;
    uint32_t dst_level;
} LCTextureCopyCommand;

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

typedef struct LCBindlessArray {
    uint64_t _0;
} LCBindlessArray;

typedef struct LCAccel {
    uint64_t _0;
} LCAccel;

typedef enum LCArgument_Tag {
    LC_ARGUMENT_BUFFER,
    LC_ARGUMENT_TEXTURE,
    LC_ARGUMENT_UNIFORM,
    LC_ARGUMENT_BINDLESS_ARRAY,
    LC_ARGUMENT_ACCEL,
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
            struct LCBindlessArray bindless_array;
        };
        struct {
            struct LCAccel accel;
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
    size_t vertex_stride;
    struct LCBuffer index_buffer;
    size_t index_buffer_offset;
    size_t index_buffer_size;
    size_t index_stride;
} LCMeshBuildCommand;

typedef struct LCProceduralPrimitive {
    uint64_t _0;
} LCProceduralPrimitive;

typedef struct LCProceduralPrimitiveBuildCommand {
    struct LCProceduralPrimitive handle;
    enum LCAccelBuildRequest request;
    struct LCBuffer aabb_buffer;
    size_t aabb_buffer_offset;
    size_t aabb_count;
} LCProceduralPrimitiveBuildCommand;

typedef struct LCAccelBuildCommand {
    struct LCAccel accel;
    enum LCAccelBuildRequest request;
    uint32_t instance_count;
    const struct LCAccelBuildModification *modifications;
    size_t modifications_count;
    bool update_instance_buffer_only;
} LCAccelBuildCommand;

typedef struct LCBindlessArrayUpdateBuffer {
    enum LCBindlessArrayUpdateOperation op;
    struct LCBuffer handle;
    size_t offset;
} LCBindlessArrayUpdateBuffer;

typedef struct LCBindlessArrayUpdateTexture {
    enum LCBindlessArrayUpdateOperation op;
    struct LCTexture handle;
    struct LCSampler sampler;
} LCBindlessArrayUpdateTexture;

typedef struct LCBindlessArrayUpdateModification {
    size_t slot;
    struct LCBindlessArrayUpdateBuffer buffer;
    struct LCBindlessArrayUpdateTexture tex2d;
    struct LCBindlessArrayUpdateTexture tex3d;
} LCBindlessArrayUpdateModification;

typedef struct LCBindlessArrayUpdateCommand {
    struct LCBindlessArray handle;
    const struct LCBindlessArrayUpdateModification *modifications;
    size_t modifications_count;
} LCBindlessArrayUpdateCommand;

typedef enum LCCommand_Tag {
    LC_COMMAND_BUFFER_UPLOAD,
    LC_COMMAND_BUFFER_DOWNLOAD,
    LC_COMMAND_BUFFER_COPY,
    LC_COMMAND_BUFFER_TO_TEXTURE_COPY,
    LC_COMMAND_TEXTURE_TO_BUFFER_COPY,
    LC_COMMAND_TEXTURE_UPLOAD,
    LC_COMMAND_TEXTURE_DOWNLOAD,
    LC_COMMAND_TEXTURE_COPY,
    LC_COMMAND_SHADER_DISPATCH,
    LC_COMMAND_MESH_BUILD,
    LC_COMMAND_PROCEDURAL_PRIMITIVE_BUILD,
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
            struct LCTextureCopyCommand texture_copy;
        };
        struct {
            struct LCShaderDispatchCommand shader_dispatch;
        };
        struct {
            struct LCMeshBuildCommand mesh_build;
        };
        struct {
            struct LCProceduralPrimitiveBuildCommand procedural_primitive_build;
        };
        struct {
            struct LCAccelBuildCommand accel_build;
        };
        struct {
            struct LCBindlessArrayUpdateCommand bindless_array_update;
        };
    };
} LCCommand;

typedef struct LCCommandList {
    const struct LCCommand *commands;
    size_t commands_count;
} LCCommandList;

typedef struct LCKernelModule {
    uint64_t ptr;
} LCKernelModule;

typedef struct LCCreatedResourceInfo {
    uint64_t handle;
    void *native_handle;
} LCCreatedResourceInfo;

typedef struct LCCreatedBufferInfo {
    struct LCCreatedResourceInfo resource;
    size_t element_stride;
    size_t total_size_bytes;
} LCCreatedBufferInfo;

typedef struct LCCreatedShaderInfo {
    struct LCCreatedResourceInfo resource;
    uint32_t block_size[3];
} LCCreatedShaderInfo;

typedef struct LCCreatedSwapchainInfo {
    struct LCCreatedResourceInfo resource;
    enum LCPixelStorage storage;
} LCCreatedSwapchainInfo;

typedef struct LCAccelOption {
    enum LCAccelUsageHint hint;
    bool allow_compaction;
    bool allow_update;
} LCAccelOption;

typedef struct LCShaderOption {
    bool enable_cache;
    bool enable_fast_math;
    bool enable_debug_info;
    bool compile_only;
    const char *name;
} LCShaderOption;

typedef void (*LCDispatchCallback)(uint8_t*);

typedef struct LCDeviceInterface {
    struct LCDevice device;
    void (*destroy_device)(struct LCDeviceInterface);
    struct LCCreatedBufferInfo (*create_buffer)(struct LCDevice, const void*, size_t);
    void (*destroy_buffer)(struct LCDevice, struct LCBuffer);
    struct LCCreatedResourceInfo (*create_texture)(struct LCDevice,
                                                   enum LCPixelFormat,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   bool);
    void *(*native_handle)(struct LCDevice);
    uint32_t (*compute_warp_size)(struct LCDevice);
    void (*destroy_texture)(struct LCDevice, struct LCTexture);
    struct LCCreatedResourceInfo (*create_bindless_array)(struct LCDevice, size_t);
    void (*destroy_bindless_array)(struct LCDevice, struct LCBindlessArray);
    struct LCCreatedResourceInfo (*create_stream)(struct LCDevice, enum LCStreamTag);
    void (*destroy_stream)(struct LCDevice, struct LCStream);
    void (*synchronize_stream)(struct LCDevice, struct LCStream);
    void (*dispatch)(struct LCDevice,
                     struct LCStream,
                     struct LCCommandList,
                     LCDispatchCallback,
                     uint8_t*);
    struct LCCreatedSwapchainInfo (*create_swapchain)(struct LCDevice,
                                                      uint64_t,
                                                      struct LCStream,
                                                      uint32_t,
                                                      uint32_t,
                                                      bool,
                                                      bool,
                                                      uint32_t);
    void (*present_display_in_stream)(struct LCDevice,
                                      struct LCStream,
                                      struct LCSwapchain,
                                      struct LCTexture);
    void (*destroy_swapchain)(struct LCDevice, struct LCSwapchain);
    struct LCCreatedShaderInfo (*create_shader)(struct LCDevice,
                                                struct LCKernelModule,
                                                const struct LCShaderOption*);
    void (*destroy_shader)(struct LCDevice, struct LCShader);
    struct LCCreatedResourceInfo (*create_event)(struct LCDevice);
    void (*destroy_event)(struct LCDevice, struct LCEvent);
    void (*signal_event)(struct LCDevice, struct LCEvent, struct LCStream, uint64_t);
    void (*synchronize_event)(struct LCDevice, struct LCEvent, uint64_t);
    void (*wait_event)(struct LCDevice, struct LCEvent, struct LCStream, uint64_t);
    bool (*is_event_completed)(struct LCDevice, struct LCEvent, uint64_t);
    struct LCCreatedResourceInfo (*create_mesh)(struct LCDevice, const struct LCAccelOption*);
    void (*destroy_mesh)(struct LCDevice, struct LCMesh);
    struct LCCreatedResourceInfo (*create_procedural_primitive)(struct LCDevice,
                                                                const struct LCAccelOption*);
    void (*destroy_procedural_primitive)(struct LCDevice, struct LCProceduralPrimitive);
    struct LCCreatedResourceInfo (*create_accel)(struct LCDevice, const struct LCAccelOption*);
    void (*destroy_accel)(struct LCDevice, struct LCAccel);
    char *(*query)(struct LCDevice, const char*);
} LCDeviceInterface;

typedef struct LCLoggerMessage {
    const char *target;
    const char *level;
    const char *message;
} LCLoggerMessage;

typedef struct LCLibInterface {
    void *inner;
    void (*set_logger_callback)(void(*)(struct LCLoggerMessage));
    struct LCContext (*create_context)(const char*);
    void (*destroy_context)(struct LCContext);
    struct LCDeviceInterface (*create_device)(struct LCContext, const char*, const char*);
    void (*free_string)(char*);
} LCLibInterface;
