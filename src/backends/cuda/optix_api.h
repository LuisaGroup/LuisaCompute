#pragma once

#include <cuda.h>
#include <cstdint>

namespace luisa::compute::optix {

constexpr auto VERSION = 70300u;

using uint = uint32_t;

using Result = uint;
using ProgramGroupKind = uint;
using CompileDebugLevel = uint;
using DeviceContext = void *;
using LogCallback = void (*)(uint, const char *, const char *, void *);
using Module = void *;
using DeviceContext = void *;
using ProgramGroup = void *;
using Pipeline = void *;
using Denoiser = void *;
using TraversableHandle = uint64_t;
using BuildOperation = uint;

constexpr auto EXCEPTION_FLAG_NONE = 0u;
constexpr auto EXCEPTION_FLAG_STACK_OVERFLOW = 1u;
constexpr auto EXCEPTION_FLAG_TRACE_DEPTH = 2u;
constexpr auto EXCEPTION_FLAG_DEBUG = 8u;

constexpr auto COMPILE_DEBUG_LEVEL_NONE = 0x2350u;
constexpr auto COMPILE_DEBUG_LEVEL_MINIMAL = 0x2351u;
constexpr auto COMPILE_OPTIMIZATION_LEVEL_0 = 0x2340u;
constexpr auto COMPILE_OPTIMIZATION_LEVEL_1 = 0x2341u;
constexpr auto COMPILE_OPTIMIZATION_LEVEL_2 = 0x2342u;
constexpr auto COMPILE_OPTIMIZATION_LEVEL_3 = 0x2343u;

constexpr auto DEVICE_CONTEXT_VALIDATION_MODE_OFF = 0u;
constexpr auto DEVICE_CONTEXT_VALIDATION_MODE_ALL = 0xFFFFFFFFu;

constexpr auto MODULE_COMPILE_STATE_COMPLETED = 0x2364u;

constexpr auto SBT_RECORD_HEADER_SIZE = 32u;
constexpr auto SBT_RECORD_ALIGNMENT = 16u;
constexpr auto ACCEL_BUFFER_BYTE_ALIGNMENT = 128u;
constexpr auto INSTANCE_BYTE_ALIGNMENT = 16u;
constexpr auto COMPILE_DEFAULT_MAX_REGISTER_COUNT = 0u;

constexpr auto BUILD_FLAG_NONE = 0u;
constexpr auto BUILD_FLAG_ALLOW_UPDATE = 1u << 0u;
constexpr auto BUILD_FLAG_ALLOW_COMPACTION = 1u << 1u;
constexpr auto BUILD_FLAG_PREFER_FAST_TRACE = 1u << 2u;
constexpr auto BUILD_FLAG_PREFER_FAST_BUILD = 1u << 3u;

constexpr auto BUILD_INPUT_TYPE_TRIANGLES = 0x2141u;
constexpr auto BUILD_INPUT_TYPE_INSTANCES = 0x2143u;

constexpr auto BUILD_OPERATION_BUILD = 0x2161u;
constexpr auto BUILD_OPERATION_UPDATE = 0x2162u;

constexpr auto PROPERTY_TYPE_COMPACTED_SIZE = 0x2181u;
constexpr auto PROPERTY_TYPE_AABBS = 0x2182u;

constexpr auto GEOMETRY_FLAG_NONE = 0u;
constexpr auto GEOMETRY_FLAG_DISABLE_ANYHIT = 1u << 0u;
constexpr auto GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1u << 1u;
constexpr auto GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 2u;

constexpr auto VERTEX_FORMAT_NONE      = 0u;
constexpr auto VERTEX_FORMAT_FLOAT3    = 0x2121u;
constexpr auto VERTEX_FORMAT_FLOAT2    = 0x2122u;
constexpr auto VERTEX_FORMAT_HALF3     = 0x2123u;
constexpr auto VERTEX_FORMAT_HALF2     = 0x2124u;
constexpr auto VERTEX_FORMAT_SNORM16_3 = 0x2125u;
constexpr auto VERTEX_FORMAT_SNORM16_2 = 0x2126u;

constexpr auto INDICES_FORMAT_NONE = 0u;
constexpr auto INDICES_FORMAT_UNSIGNED_SHORT3 = 0x2102u;
constexpr auto INDICES_FORMAT_UNSIGNED_INT3 = 0x2103u;

constexpr auto PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1u << 31u;

constexpr auto TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0u;
constexpr auto TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0u;
constexpr auto TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1u;

constexpr auto PROGRAM_GROUP_KIND_RAYGEN = 0x2421u;
constexpr auto PROGRAM_GROUP_KIND_MISS = 0x2422u;
constexpr auto PROGRAM_GROUP_KIND_EXCEPTION = 0x2423u;
constexpr auto PROGRAM_GROUP_KIND_HITGROUP = 0x2424u;
constexpr auto PROGRAM_GROUP_KIND_CALLABLES = 0x2425u;

struct PipelineCompileOptions {
    int usesMotionBlur;
    uint traversableGraphFlags;
    uint numPayloadValues;
    uint numAttributeValues;
    uint exceptionFlags;
    const char *pipelineLaunchParamsVariableName;
    uint usesPrimitiveTypeFlags;
    uint reserved;
    size_t reserved2;
};

struct ShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr missRecordBase;
    uint missRecordStrideInBytes;
    uint missRecordCount;
    CUdeviceptr hitgroupRecordBase;
    uint hitgroupRecordStrideInBytes;
    uint hitgroupRecordCount;
    CUdeviceptr callablesRecordBase;
    uint callablesRecordStrideInBytes;
    uint callablesRecordCount;
};

struct DeviceContextOptions {
    LogCallback logCallbackFunction;
    void *logCallbackData;
    uint logCallbackLevel;
    uint validationMode;
};

struct PayloadType {
    uint numPayloadValues;
    const uint *payloadSemantics;
};

struct ModuleCompileOptions {
    uint maxRegisterCount;
    uint optLevel;
    uint debugLevel;
    const void *boundValues;
    uint numBoundValues;
    uint numPayloadTypes;
    PayloadType *payloadTypes;
};

struct PipelineLinkOptions {
    uint maxTraceDepth;
    CompileDebugLevel debugLevel;
};

struct ProgramGroupSingleModule {
    Module module;
    const char *entryFunctionName;
};

struct ProgramGroupHitgroup {
    Module moduleCH;
    const char *entryFunctionNameCH;
    Module moduleAH;
    const char *entryFunctionNameAH;
    Module moduleIS;
    const char *entryFunctionNameIS;
};

struct ProgramGroupCallables {
    Module moduleDC;
    const char *entryFunctionNameDC;
    Module moduleCC;
    const char *entryFunctionNameCC;
};

struct ProgramGroupDesc {
    ProgramGroupKind kind;
    uint flags;
    union {
        ProgramGroupSingleModule raygen;
        ProgramGroupSingleModule miss;
        ProgramGroupSingleModule exception;
        ProgramGroupCallables callables;
        ProgramGroupHitgroup hitgroup;
    };
};

struct StackSizes {
    uint cssRG;
    uint cssMS;
    uint cssCH;
    uint cssAH;
    uint cssIS;
    uint cssCC;
    uint dssDC;
};

struct ProgramGroupOptions {
    PayloadType *payloadType;
};

struct BuiltinISOptions {
    uint builtinISModuleType;
    int usesMotionBlur;
};

struct MotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct AccelBuildOptions {
    uint buildFlags;
    uint operation;
    MotionOptions motionOptions;
};

struct alignas(16) Instance {
    float transform[12];
    uint instanceId;
    uint sbtOffset;
    uint visibilityMask;
    uint flags;
    TraversableHandle handle;
    uint pad[2];
};

static_assert(sizeof(Instance) == 80u);

struct BuildInputTriangleArray {
    const CUdeviceptr *vertexBuffers;
    uint numVertices;
    uint vertexFormat;
    uint vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    uint numIndexTriplets;
    uint indexFormat;
    uint indexStrideInBytes;
    CUdeviceptr preTransform;
    const uint *flags;
    uint numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    uint sbtIndexOffsetSizeInBytes;
    uint sbtIndexOffsetStrideInBytes;
    uint primitiveIndexOffset;
    uint transformFormat;
};

struct BuildInputInstanceArray {
    CUdeviceptr instances;
    uint numInstances;
};

struct BuildInput {
    uint type;
    union {
        BuildInputTriangleArray triangleArray;
        BuildInputInstanceArray instanceArray;
        uint8_t pad[1024];
    };
};

struct AccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct AccelEmitDesc {
    CUdeviceptr result;
    uint type;
};

struct AccelRelocationInfo {
    uint64_t info[4];
};

struct DenoiserOptions {
    uint guideAlbedo;
    uint guideNormal;
};

struct DenoiserSizes {
    size_t stateSizeInBytes;
    size_t withOverlapScratchSizeInBytes;
    size_t withoutOverlapScratchSizeInBytes;
    uint overlapWindowSizeInPixels;
};

struct DenoiserParams {
    uint denoiseAlpha;
    CUdeviceptr hdrIntensity;
    float blendFactor;
    CUdeviceptr hdrAverageColor;
};

struct Image2D {
    CUdeviceptr data;
    uint width;
    uint height;
    uint rowStrideInBytes;
    uint pixelStrideInBytes;
    uint format;
};

struct DenoiserGuideLayer {
    Image2D albedo;
    Image2D normal;
    Image2D flow;
};

struct DenoiserLayer {
    Image2D input;
    Image2D previousOutput;
    Image2D output;
};

struct FunctionTable {
    const char *(*getErrorName)(Result result);
    const char *(*getErrorString)(Result result);
    Result (*deviceContextCreate)(CUcontext fromContext, const DeviceContextOptions *options, DeviceContext *context);
    Result (*deviceContextDestroy)(DeviceContext context);
    Result (*deviceContextGetProperty)(DeviceContext context, uint property, void *value, size_t sizeInBytes);
    Result (*deviceContextSetLogCallback)(DeviceContext context,
                                          LogCallback callbackFunction,
                                          void *callbackData,
                                          uint callbackLevel);
    Result (*deviceContextSetCacheEnabled)(DeviceContext context, int enabled);
    Result (*deviceContextSetCacheLocation)(DeviceContext context, const char *location);
    Result (*deviceContextSetCacheDatabaseSizes)(DeviceContext context, size_t lowWaterMark, size_t highWaterMark);
    Result (*deviceContextGetCacheEnabled)(DeviceContext context, int *enabled);
    Result (*deviceContextGetCacheLocation)(DeviceContext context, char *location, size_t locationSize);
    Result (*deviceContextGetCacheDatabaseSizes)(DeviceContext context, size_t *lowWaterMark, size_t *highWaterMark);
    Result (*moduleCreateFromPTX)(DeviceContext context,
                                  const ModuleCompileOptions *moduleCompileOptions,
                                  const PipelineCompileOptions *pipelineCompileOptions,
                                  const char *PTX,
                                  size_t PTXsize,
                                  char *logString,
                                  size_t *logStringSize,
                                  Module *module);
    Result (*moduleDestroy)(Module module);
    Result (*builtinISModuleGet)(DeviceContext context,
                                 const ModuleCompileOptions *moduleCompileOptions,
                                 const PipelineCompileOptions *pipelineCompileOptions,
                                 const BuiltinISOptions *builtinISOptions,
                                 Module *builtinModule);
    Result (*programGroupCreate)(DeviceContext context,
                                 const ProgramGroupDesc *programDescriptions,
                                 uint numProgramGroups,
                                 const ProgramGroupOptions *options,
                                 char *logString,
                                 size_t *logStringSize,
                                 ProgramGroup *programGroups);
    Result (*programGroupDestroy)(ProgramGroup programGroup);
    Result (*programGroupGetStackSize)(ProgramGroup programGroup, StackSizes *stackSizes);
    Result (*pipelineCreate)(DeviceContext context,
                             const PipelineCompileOptions *pipelineCompileOptions,
                             const PipelineLinkOptions *pipelineLinkOptions,
                             const ProgramGroup *programGroups,
                             uint numProgramGroups,
                             char *logString,
                             size_t *logStringSize,
                             Pipeline *pipeline);
    Result (*pipelineDestroy)(Pipeline pipeline);
    Result (*pipelineSetStackSize)(Pipeline pipeline,
                                   uint directCallableStackSizeFromTraversal,
                                   uint directCallableStackSizeFromState,
                                   uint continuationStackSize,
                                   uint maxTraversableGraphDepth);
    Result (*accelComputeMemoryUsage)(DeviceContext context,
                                      const AccelBuildOptions *accelOptions,
                                      const BuildInput *buildInputs,
                                      uint numBuildInputs,
                                      AccelBufferSizes *bufferSizes);
    Result (*accelBuild)(DeviceContext context,
                         CUstream stream,
                         const AccelBuildOptions *accelOptions,
                         const BuildInput *buildInputs,
                         uint numBuildInputs,
                         CUdeviceptr tempBuffer,
                         size_t tempBufferSizeInBytes,
                         CUdeviceptr outputBuffer,
                         size_t outputBufferSizeInBytes,
                         uint64_t *outputHandle,
                         const AccelEmitDesc *emittedProperties,
                         uint numEmittedProperties);
    Result (*accelGetRelocationInfo)(DeviceContext context, TraversableHandle handle, AccelRelocationInfo *info);

    Result (*accelCheckRelocationCompatibility)(DeviceContext context,
                                                const AccelRelocationInfo *info,
                                                int *compatible);
    Result (*accelRelocate)(DeviceContext context,
                            CUstream stream,
                            const AccelRelocationInfo *info,
                            CUdeviceptr instanceTraversableHandles,
                            size_t numInstanceTraversableHandles,
                            CUdeviceptr targetAccel,
                            size_t targetAccelSizeInBytes,
                            uint64_t *targetHandle);
    Result (*accelCompact)(DeviceContext context,
                           CUstream stream,
                           uint64_t inputHandle,
                           CUdeviceptr outputBuffer,
                           size_t outputBufferSizeInBytes,
                           uint64_t *outputHandle);
    Result (*convertPointerToTraversableHandle)(DeviceContext onDevice,
                                                CUdeviceptr pointer,
                                                uint traversableType,
                                                uint64_t *traversableHandle);
    Result (*sbtRecordPackHeader)(ProgramGroup programGroup, void *sbtRecordHeaderHostPointer);
    Result (*launch)(Pipeline pipeline,
                     CUstream stream,
                     CUdeviceptr pipelineParams,
                     size_t pipelineParamsSize,
                     const ShaderBindingTable *sbt,
                     uint width,
                     uint height,
                     uint depth);
    Result (*denoiserCreate)(DeviceContext context, uint modelKind, const DenoiserOptions *options, Denoiser *returnHandle);
    Result (*denoiserDestroy)(Denoiser handle);
    Result (*denoiserComputeMemoryResources)(Denoiser handle,
                                             uint maximumInputWidth,
                                             uint maximumInputHeight,
                                             DenoiserSizes *returnSizes);
    Result (*denoiserSetup)(Denoiser denoiser,
                            CUstream stream,
                            uint inputWidth,
                            uint inputHeight,
                            CUdeviceptr state,
                            size_t stateSizeInBytes,
                            CUdeviceptr scratch,
                            size_t scratchSizeInBytes);
    Result (*denoiserInvoke)(Denoiser denoiser,
                             CUstream stream,
                             const DenoiserParams *params,
                             CUdeviceptr denoiserState,
                             size_t denoiserStateSizeInBytes,
                             const DenoiserGuideLayer *guideLayer,
                             const DenoiserLayer *layers,
                             uint numLayers,
                             uint inputOffsetX,
                             uint inputOffsetY,
                             CUdeviceptr scratch,
                             size_t scratchSizeInBytes);
    Result (*denoiserComputeIntensity)(Denoiser handle,
                                       CUstream stream,
                                       const Image2D *inputImage,
                                       CUdeviceptr outputIntensity,
                                       CUdeviceptr scratch,
                                       size_t scratchSizeInBytes);
    Result (*denoiserComputeAverageColor)(Denoiser handle,
                                          CUstream stream,
                                          const Image2D *inputImage,
                                          CUdeviceptr outputAverageColor,
                                          CUdeviceptr scratch,
                                          size_t scratchSizeInBytes);
    Result (*denoiserCreateWithUserModel)(DeviceContext context, const void *data,
                                          size_t dataSizeInBytes, Denoiser *returnHandle);
};

[[nodiscard]] const FunctionTable &api() noexcept;

}// namespace luisa::compute::optix
