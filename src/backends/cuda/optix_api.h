#pragma once

#include <cuda.h>
#include <cstdint>
#include <stdint.h>

namespace luisa::compute::optix {

using OptixResult = int;
using OptixProgramGroupKind = int;
using OptixCompileDebugLevel = int;
using OptixDeviceContext = void *;
using OptixLogCallback = void (*)(unsigned int, const char *, const char *, void *);
using OptixTask = void *;
using OptixModule = void *;
using OptixDeviceContext = void *;
using OptixProgramGroup = void *;
using OptixPipeline = void *;
using OptixDenoiser = void *;

#define OPTIX_EXCEPTION_FLAG_NONE 0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW 1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH 2
#define OPTIX_EXCEPTION_FLAG_DEBUG 8
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE 0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL 0x2351
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 0x2340
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 0x2341
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 0x2342
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 0x2343
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF 0
#define OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ((int)0xFFFFFFFF)
#define OPTIX_MODULE_COMPILE_STATE_COMPLETED 0x2364
#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN 0x2421
#define OPTIX_PROGRAM_GROUP_KIND_CALLABLES 0x2425
#define OPTIX_PROGRAM_GROUP_KIND_MISS 0x2422
#define OPTIX_SBT_RECORD_HEADER_SIZE 32
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char *pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
    unsigned int reserved;
    size_t reserved2;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

struct OptixDeviceContextOptions {
    OptixLogCallback logCallbackFunction;
    void *logCallbackData;
    int logCallbackLevel;
    int validationMode;
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    OptixPayloadType *payloadTypes;
};

struct OptixPipelineLinkOptions {
    unsigned int maxTraceDepth;
    OptixCompileDebugLevel debugLevel;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char *entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char *entryFunctionNameCH;
    OptixModule moduleAH;
    const char *entryFunctionNameAH;
    OptixModule moduleIS;
    const char *entryFunctionNameIS;
};

struct OptixProgramGroupCallables {
    OptixModule moduleDC;
    const char *entryFunctionNameDC;
    OptixModule moduleCC;
    const char *entryFunctionNameCC;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupCallables callables;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixStackSizes {
    unsigned int cssRG;
    unsigned int cssMS;
    unsigned int cssCH;
    unsigned int cssAH;
    unsigned int cssIS;
    unsigned int cssCC;
    unsigned int dssDC;
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

struct OptixBuiltinISOptions {
    uint32_t builtinISModuleType;
    int usesMotionBlur;
};

struct OptixMotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct OptixAccelBuildOptions {
    unsigned int buildFlags;
    uint32_t operation;
    OptixMotionOptions motionOptions;
};

struct OptixBuildInputTriangleArray {
    const CUdeviceptr *vertexBuffers;
    unsigned int numVertices;
    uint32_t vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    uint32_t indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int *flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    uint32_t transformFormat;
};

struct OptixBuildInputInstanceArray {
    CUdeviceptr instances;
    unsigned int numInstances;
};

struct OptixBuildInput {
    uint32_t type;
    union {
        OptixBuildInputTriangleArray triangleArray;
        OptixBuildInputInstanceArray instanceArray;
        char pad[1024];
    };
};

struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct OptixAccelEmitDesc {
    CUdeviceptr result;
    uint32_t type;
};

struct OptixAccelRelocationInfo {
    unsigned long long info[4];
};

struct OptixDenoiserOptions {
    unsigned int guideAlbedo;
    unsigned int guideNormal;
};

struct OptixDenoiserSizes {
    size_t stateSizeInBytes;
    size_t withOverlapScratchSizeInBytes;
    size_t withoutOverlapScratchSizeInBytes;
    unsigned int overlapWindowSizeInPixels;
};

struct OptixDenoiserParams {
    unsigned int denoiseAlpha;
    CUdeviceptr hdrIntensity;
    float blendFactor;
    CUdeviceptr hdrAverageColor;
};

struct OptixImage2D {
    CUdeviceptr data;
    unsigned int width;
    unsigned int height;
    unsigned int rowStrideInBytes;
    unsigned int pixelStrideInBytes;
    uint32_t format;
};

struct OptixDenoiserGuideLayer {
    OptixImage2D albedo;
    OptixImage2D normal;
    OptixImage2D flow;
};

struct OptixDenoiserLayer {
    OptixImage2D input;
    OptixImage2D previousOutput;
    OptixImage2D output;
};

struct OptixFunctionTable {
    const char *(*getErrorName)(OptixResult result);
    const char *(*getErrorString)(OptixResult result);
    OptixResult (*deviceContextCreate)(CUcontext fromContext, const OptixDeviceContextOptions *options, OptixDeviceContext *context);
    OptixResult (*deviceContextDestroy)(OptixDeviceContext context);
    OptixResult (*deviceContextGetProperty)(OptixDeviceContext context, uint32_t property, void *value, size_t sizeInBytes);
    OptixResult (*deviceContextSetLogCallback)(OptixDeviceContext context,
                                               OptixLogCallback callbackFunction,
                                               void *callbackData,
                                               unsigned int callbackLevel);
    OptixResult (*deviceContextSetCacheEnabled)(OptixDeviceContext context, int enabled);
    OptixResult (*deviceContextSetCacheLocation)(OptixDeviceContext context, const char *location);
    OptixResult (*deviceContextSetCacheDatabaseSizes)(OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark);
    OptixResult (*deviceContextGetCacheEnabled)(OptixDeviceContext context, int *enabled);
    OptixResult (*deviceContextGetCacheLocation)(OptixDeviceContext context, char *location, size_t locationSize);
    OptixResult (*deviceContextGetCacheDatabaseSizes)(OptixDeviceContext context, size_t *lowWaterMark, size_t *highWaterMark);
    OptixResult (*moduleCreateFromPTX)(OptixDeviceContext context,
                                       const OptixModuleCompileOptions *moduleCompileOptions,
                                       const OptixPipelineCompileOptions *pipelineCompileOptions,
                                       const char *PTX,
                                       size_t PTXsize,
                                       char *logString,
                                       size_t *logStringSize,
                                       OptixModule *module);
    OptixResult (*moduleDestroy)(OptixModule module);
    OptixResult (*builtinISModuleGet)(OptixDeviceContext context,
                                      const OptixModuleCompileOptions *moduleCompileOptions,
                                      const OptixPipelineCompileOptions *pipelineCompileOptions,
                                      const OptixBuiltinISOptions *builtinISOptions,
                                      OptixModule *builtinModule);
    OptixResult (*programGroupCreate)(OptixDeviceContext context,
                                      const OptixProgramGroupDesc *programDescriptions,
                                      unsigned int numProgramGroups,
                                      const OptixProgramGroupOptions *options,
                                      char *logString,
                                      size_t *logStringSize,
                                      OptixProgramGroup *programGroups);
    OptixResult (*programGroupDestroy)(OptixProgramGroup programGroup);
    OptixResult (*programGroupGetStackSize)(OptixProgramGroup programGroup, OptixStackSizes *stackSizes);
    OptixResult (*pipelineCreate)(OptixDeviceContext context,
                                  const OptixPipelineCompileOptions *pipelineCompileOptions,
                                  const OptixPipelineLinkOptions *pipelineLinkOptions,
                                  const OptixProgramGroup *programGroups,
                                  unsigned int numProgramGroups,
                                  char *logString,
                                  size_t *logStringSize,
                                  OptixPipeline *pipeline);
    OptixResult (*pipelineDestroy)(OptixPipeline pipeline);
    OptixResult (*pipelineSetStackSize)(OptixPipeline pipeline,
                                        unsigned int directCallableStackSizeFromTraversal,
                                        unsigned int directCallableStackSizeFromState,
                                        unsigned int continuationStackSize,
                                        unsigned int maxTraversableGraphDepth);
    OptixResult (*accelComputeMemoryUsage)(OptixDeviceContext context,
                                           const OptixAccelBuildOptions *accelOptions,
                                           const OptixBuildInput *buildInputs,
                                           unsigned int numBuildInputs,
                                           OptixAccelBufferSizes *bufferSizes);
    OptixResult (*accelBuild)(OptixDeviceContext context,
                              CUstream stream,
                              const OptixAccelBuildOptions *accelOptions,
                              const OptixBuildInput *buildInputs,
                              unsigned int numBuildInputs,
                              CUdeviceptr tempBuffer,
                              size_t tempBufferSizeInBytes,
                              CUdeviceptr outputBuffer,
                              size_t outputBufferSizeInBytes,
                              uint64_t *outputHandle,
                              const OptixAccelEmitDesc *emittedProperties,
                              unsigned int numEmittedProperties);
    OptixResult (*accelGetRelocationInfo)(OptixDeviceContext context, uint64_t handle, OptixAccelRelocationInfo *info);

    OptixResult (*accelCheckRelocationCompatibility)(OptixDeviceContext context,
                                                     const OptixAccelRelocationInfo *info,
                                                     int *compatible);
    OptixResult (*accelRelocate)(OptixDeviceContext context,
                                 CUstream stream,
                                 const OptixAccelRelocationInfo *info,
                                 CUdeviceptr instanceTraversableHandles,
                                 size_t numInstanceTraversableHandles,
                                 CUdeviceptr targetAccel,
                                 size_t targetAccelSizeInBytes,
                                 uint64_t *targetHandle);
    OptixResult (*accelCompact)(OptixDeviceContext context,
                                CUstream stream,
                                uint64_t inputHandle,
                                CUdeviceptr outputBuffer,
                                size_t outputBufferSizeInBytes,
                                uint64_t *outputHandle);
    OptixResult (*convertPointerToTraversableHandle)(OptixDeviceContext onDevice,
                                                     CUdeviceptr pointer,
                                                     uint32_t traversableType,
                                                     uint64_t *traversableHandle);
    OptixResult (*sbtRecordPackHeader)(OptixProgramGroup programGroup, void *sbtRecordHeaderHostPointer);
    OptixResult (*launch)(OptixPipeline pipeline,
                          CUstream stream,
                          CUdeviceptr pipelineParams,
                          size_t pipelineParamsSize,
                          const OptixShaderBindingTable *sbt,
                          unsigned int width,
                          unsigned int height,
                          unsigned int depth);
    OptixResult (*denoiserCreate)(OptixDeviceContext context, uint32_t modelKind, const OptixDenoiserOptions *options, OptixDenoiser *returnHandle);
    OptixResult (*denoiserDestroy)(OptixDenoiser handle);
    OptixResult (*denoiserComputeMemoryResources)(const OptixDenoiser handle,
                                                  unsigned int maximumInputWidth,
                                                  unsigned int maximumInputHeight,
                                                  OptixDenoiserSizes *returnSizes);
    OptixResult (*denoiserSetup)(OptixDenoiser denoiser,
                                 CUstream stream,
                                 unsigned int inputWidth,
                                 unsigned int inputHeight,
                                 CUdeviceptr state,
                                 size_t stateSizeInBytes,
                                 CUdeviceptr scratch,
                                 size_t scratchSizeInBytes);
    OptixResult (*denoiserInvoke)(OptixDenoiser denoiser,
                                  CUstream stream,
                                  const OptixDenoiserParams *params,
                                  CUdeviceptr denoiserState,
                                  size_t denoiserStateSizeInBytes,
                                  const OptixDenoiserGuideLayer *guideLayer,
                                  const OptixDenoiserLayer *layers,
                                  unsigned int numLayers,
                                  unsigned int inputOffsetX,
                                  unsigned int inputOffsetY,
                                  CUdeviceptr scratch,
                                  size_t scratchSizeInBytes);
    OptixResult (*denoiserComputeIntensity)(OptixDenoiser handle,
                                            CUstream stream,
                                            const OptixImage2D *inputImage,
                                            CUdeviceptr outputIntensity,
                                            CUdeviceptr scratch,
                                            size_t scratchSizeInBytes);
    OptixResult (*denoiserComputeAverageColor)(OptixDenoiser handle,
                                               CUstream stream,
                                               const OptixImage2D *inputImage,
                                               CUdeviceptr outputAverageColor,
                                               CUdeviceptr scratch,
                                               size_t scratchSizeInBytes);
    OptixResult (*denoiserCreateWithUserModel)(OptixDeviceContext context, const void *data, size_t dataSizeInBytes, OptixDenoiser *returnHandle);
};

[[nodiscard]] const OptixFunctionTable &api() noexcept;

}// namespace luisa::compute::optix
