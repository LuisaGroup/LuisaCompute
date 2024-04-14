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

typedef enum LCCurveBasis {
    LC_CURVE_BASIS_PIECEWISE_LINEAR = 0,
    LC_CURVE_BASIS_CUBIC_B_SPLINE = 1,
    LC_CURVE_BASIS_CATMULL_ROM = 2,
    LC_CURVE_BASIS_BEZIER = 3,
} LCCurveBasis;

typedef enum LCFilterQuality {
    LC_FILTER_QUALITY_DEFAULT,
    LC_FILTER_QUALITY_FAST,
    LC_FILTER_QUALITY_ACCURATE,
} LCFilterQuality;

typedef enum LCImageColorSpace {
    LC_IMAGE_COLOR_SPACE_HDR,
    LC_IMAGE_COLOR_SPACE_LDR_LINEAR,
    LC_IMAGE_COLOR_SPACE_LDR_SRGB,
} LCImageColorSpace;

typedef enum LCImageFormat {
    LC_IMAGE_FORMAT_FLOAT1,
    LC_IMAGE_FORMAT_FLOAT2,
    LC_IMAGE_FORMAT_FLOAT3,
    LC_IMAGE_FORMAT_FLOAT4,
    LC_IMAGE_FORMAT_HALF1,
    LC_IMAGE_FORMAT_HALF2,
    LC_IMAGE_FORMAT_HALF3,
    LC_IMAGE_FORMAT_HALF4,
} LCImageFormat;

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
    LC_PIXEL_FORMAT_R10G10B10A2U_INT,
    LC_PIXEL_FORMAT_R10G10B10A2U_NORM,
    LC_PIXEL_FORMAT_R11G11B10F,
    LC_PIXEL_FORMAT_BC1U_NORM,
    LC_PIXEL_FORMAT_BC2U_NORM,
    LC_PIXEL_FORMAT_BC3U_NORM,
    LC_PIXEL_FORMAT_BC4U_NORM,
    LC_PIXEL_FORMAT_BC5U_NORM,
    LC_PIXEL_FORMAT_BC6HUF16,
    LC_PIXEL_FORMAT_BC7U_NORM,
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
    LC_PIXEL_STORAGE_R10G10B10A2,
    LC_PIXEL_STORAGE_R11G11B10,
    LC_PIXEL_STORAGE_BC1,
    LC_PIXEL_STORAGE_BC2,
    LC_PIXEL_STORAGE_BC3,
    LC_PIXEL_STORAGE_BC4,
    LC_PIXEL_STORAGE_BC5,
    LC_PIXEL_STORAGE_BC6,
    LC_PIXEL_STORAGE_BC7,
} LCPixelStorage;

typedef enum LCPrefilterMode {
    LC_PREFILTER_MODE_NONE,
    LC_PREFILTER_MODE_FAST,
    LC_PREFILTER_MODE_ACCURATE,
} LCPrefilterMode;

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

typedef struct LCCurve {
    uint64_t _0;
} LCCurve;

typedef struct LCCurveBuildCommand {
    struct LCCurve curve;
    enum LCAccelBuildRequest request;
    enum LCCurveBasis basis;
    size_t cp_count;
    size_t seg_count;
    struct LCBuffer cp_buffer;
    size_t cp_buffer_offset;
    size_t cp_buffer_stride;
    struct LCBuffer seg_buffer;
    size_t seg_buffer_offset;
} LCCurveBuildCommand;

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
    LC_COMMAND_CURVE_BUILD,
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
            struct LCCurveBuildCommand curve_build;
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
    bool time_trace;
    uint32_t max_registers;
    const char *name;
    const char *native_include;
} LCShaderOption;

typedef void (*LCDispatchCallback)(uint8_t*);

typedef struct LCSwapchainOption {
    uint64_t display;
    uint64_t window;
    uint32_t width;
    uint32_t height;
    bool wants_hdr;
    bool wants_vsync;
    uint32_t back_buffer_count;
} LCSwapchainOption;

typedef struct LCPinnedMemoryOption {
    bool write_combined;
} LCPinnedMemoryOption;

typedef struct LCPinnedMemoryExt {
    void *data;
    void (*pin_host_memory)(struct LCPinnedMemoryExt*,
                            const void*,
                            size_t,
                            void*,
                            const struct LCPinnedMemoryOption*);
    void (*allocate_pinned_memory)(struct LCPinnedMemoryExt*, size_t, void*);
} LCPinnedMemoryExt;

typedef struct LCDenoiser {
    uint8_t _unused[0];
} LCDenoiser;

typedef struct LCImage {
    enum LCImageFormat format;
    uint64_t buffer_handle;
    void *device_ptr;
    size_t offset;
    size_t pixel_stride;
    size_t row_stride;
    size_t size_bytes;
    enum LCImageColorSpace color_space;
    float input_scale;
} LCImage;

typedef struct LCFeature {
    const char *name;
    size_t name_len;
    struct LCImage image;
} LCFeature;

typedef struct LCDenoiserInput {
    const struct LCImage *inputs;
    size_t inputs_count;
    const struct LCImage *outputs;
    const struct LCFeature *features;
    size_t features_count;
    enum LCPrefilterMode prefilter_mode;
    enum LCFilterQuality filter_quality;
    bool noisy_features;
    uint32_t width;
    uint32_t height;
} LCDenoiserInput;

typedef struct LCDenoiserExt {
    void *data;
    struct LCDenoiser *(*create)(const struct LCDenoiserExt*, uint64_t stream);
    void (*init)(const struct LCDenoiserExt*, struct LCDenoiser*, const struct LCDenoiserInput*);
    void (*execute)(const struct LCDenoiserExt*, struct LCDenoiser*, bool);
    void (*destroy)(const struct LCDenoiserExt*, struct LCDenoiser*);
} LCDenoiserExt;

typedef struct LCDeviceInterface {
    struct LCDevice device;
    void (*destroy_device)(struct LCDeviceInterface);
    struct LCCreatedBufferInfo (*create_buffer)(struct LCDevice, const void*, size_t, void*);
    void (*destroy_buffer)(struct LCDevice, struct LCBuffer);
    struct LCCreatedResourceInfo (*create_texture)(struct LCDevice,
                                                   enum LCPixelFormat,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   bool,
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
                                                      const struct LCSwapchainOption*,
                                                      struct LCStream);
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
    struct LCCreatedResourceInfo (*create_curve)(struct LCDevice, const struct LCAccelOption*);
    void (*destroy_curve)(struct LCDevice, struct LCCurve);
    struct LCCreatedResourceInfo (*create_procedural_primitive)(struct LCDevice,
                                                                const struct LCAccelOption*);
    void (*destroy_procedural_primitive)(struct LCDevice, struct LCProceduralPrimitive);
    struct LCCreatedResourceInfo (*create_accel)(struct LCDevice, const struct LCAccelOption*);
    void (*destroy_accel)(struct LCDevice, struct LCAccel);
    char *(*query)(struct LCDevice, const char*);
    struct LCPinnedMemoryExt (*pinned_memory_ext)(struct LCDevice);
    struct LCDenoiserExt (*denoiser_ext)(struct LCDevice);
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
