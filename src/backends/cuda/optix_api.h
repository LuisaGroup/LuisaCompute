#pragma once

#include <array>
#include <cuda.h>

namespace luisa::compute::optix {

// versions
static constexpr auto VERSION = 80000u;
static constexpr auto ABI_VERSION = 87u;

// types
using TraversableHandle = unsigned long long;
using VisibilityMask = unsigned int;

using LogCallback = void (*)(unsigned int level, const char* tag, const char* message, void* cbdata);

using DeviceContext = struct DeviceContext_t *;
using Module = struct Module_t *;
using ProgramGroup = struct ProgramGroup_t *;
using Pipeline = struct Pipeline_t *;
using Denoiser = struct Denoiser_t *;
using Task = struct Task_t *;
static constexpr auto SBT_RECORD_HEADER_SIZE = static_cast<size_t>(32);
static constexpr auto SBT_RECORD_ALIGNMENT = 16ull;
static constexpr auto ACCEL_BUFFER_BYTE_ALIGNMENT = 128ull;
static constexpr auto INSTANCE_BYTE_ALIGNMENT = 16ull;
static constexpr auto AABB_BUFFER_BYTE_ALIGNMENT = 8ull;
static constexpr auto GEOMETRY_TRANSFORM_BYTE_ALIGNMENT = 16ull;
static constexpr auto TRANSFORM_BYTE_ALIGNMENT = 64ull;
static constexpr auto OPACITY_MICROMAP_DESC_BUFFER_BYTE_ALIGNMENT = 8ull;
static constexpr auto COMPILE_DEFAULT_MAX_REGISTER_COUNT = 0u;
static constexpr auto COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT = 8u;
static constexpr auto COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT = 32u;
static constexpr auto OPACITY_MICROMAP_STATE_TRANSPARENT = (0);
static constexpr auto OPACITY_MICROMAP_STATE_OPAQUE = (1);
static constexpr auto OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT = (2);
static constexpr auto OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE = (3);
static constexpr auto OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT = (-1);
static constexpr auto OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE = (-2);
static constexpr auto OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_TRANSPARENT = (-3);
static constexpr auto OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE = (-4);
static constexpr auto OPACITY_MICROMAP_ARRAY_BUFFER_BYTE_ALIGNMENT = 128ull;
static constexpr auto OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL = 12u;
static constexpr auto DISPLACEMENT_MICROMAP_MAX_SUBDIVISION_LEVEL = 5u;
static constexpr auto DISPLACEMENT_MICROMAP_DESC_BUFFER_BYTE_ALIGNMENT = 8ull;
static constexpr auto DISPLACEMENT_MICROMAP_ARRAY_BUFFER_BYTE_ALIGNMENT = 128ull;

enum Result : unsigned int {
    RESULT_SUCCESS = 0u,
    RESULT_ERROR_INVALID_VALUE = 7001u,
    RESULT_ERROR_HOST_OUT_OF_MEMORY = 7002u,
    RESULT_ERROR_INVALID_OPERATION = 7003u,
    RESULT_ERROR_FILE_IO_ERROR = 7004u,
    RESULT_ERROR_INVALID_FILE_FORMAT = 7005u,
    RESULT_ERROR_DISK_CACHE_INVALID_PATH = 7010u,
    RESULT_ERROR_DISK_CACHE_PERMISSION_ERROR = 7011u,
    RESULT_ERROR_DISK_CACHE_DATABASE_ERROR = 7012u,
    RESULT_ERROR_DISK_CACHE_INVALID_DATA = 7013u,
    RESULT_ERROR_LAUNCH_FAILURE = 7050u,
    RESULT_ERROR_INVALID_DEVICE_CONTEXT = 7051u,
    RESULT_ERROR_CUDA_NOT_INITIALIZED = 7052u,
    RESULT_ERROR_VALIDATION_FAILURE = 7053u,
    RESULT_ERROR_INVALID_INPUT = 7200u,
    RESULT_ERROR_INVALID_LAUNCH_PARAMETER = 7201u,
    RESULT_ERROR_INVALID_PAYLOAD_ACCESS = 7202u,
    RESULT_ERROR_INVALID_ATTRIBUTE_ACCESS = 7203u,
    RESULT_ERROR_INVALID_FUNCTION_USE = 7204u,
    RESULT_ERROR_INVALID_FUNCTION_ARGUMENTS = 7205u,
    RESULT_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY = 7250u,
    RESULT_ERROR_PIPELINE_LINK_ERROR = 7251u,
    RESULT_ERROR_ILLEGAL_DURING_TASK_EXECUTE = 7270u,
    RESULT_ERROR_INTERNAL_COMPILER_ERROR = 7299u,
    RESULT_ERROR_DENOISER_MODEL_NOT_SET = 7300u,
    RESULT_ERROR_DENOISER_NOT_INITIALIZED = 7301u,
    RESULT_ERROR_NOT_COMPATIBLE = 7400u,
    RESULT_ERROR_PAYLOAD_TYPE_MISMATCH = 7500u,
    RESULT_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED = 7501u,
    RESULT_ERROR_PAYLOAD_TYPE_ID_INVALID = 7502u,
    RESULT_ERROR_NOT_SUPPORTED = 7800u,
    RESULT_ERROR_UNSUPPORTED_ABI_VERSION = 7801u,
    RESULT_ERROR_FUNCTION_TABLE_SIZE_MISMATCH = 7802u,
    RESULT_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS = 7803u,
    RESULT_ERROR_LIBRARY_NOT_FOUND = 7804u,
    RESULT_ERROR_ENTRY_SYMBOL_NOT_FOUND = 7805u,
    RESULT_ERROR_LIBRARY_UNLOAD_FAILURE = 7806u,
    RESULT_ERROR_DEVICE_OUT_OF_MEMORY = 7807u,
    RESULT_ERROR_CUDA_ERROR = 7900u,
    RESULT_ERROR_INTERNAL_ERROR = 7990u,
    RESULT_ERROR_UNKNOWN = 7999u,
};

enum DeviceProperty : unsigned int {
    DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH = 0x2001u,
    DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH = 0x2002u,
    DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS = 0x2003u,
    DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS = 0x2004u,
    DEVICE_PROPERTY_RTCORE_VERSION = 0x2005u,
    DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID = 0x2006u,
    DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK = 0x2007u,
    DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS = 0x2008u,
    DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET = 0x2009u,
    DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING = 0x200au,
};

enum DeviceContextValidationMode : unsigned int {
    DEVICE_CONTEXT_VALIDATION_MODE_OFF = 0u,
    DEVICE_CONTEXT_VALIDATION_MODE_ALL = 0xffffffffu,
};

struct DeviceContextOptions {
    LogCallback logCallbackFunction;
    void* logCallbackData;
    int logCallbackLevel;
    DeviceContextValidationMode validationMode;
};

enum DevicePropertyShaderExecutionReorderingFlags : unsigned int {
    DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING_FLAG_NONE = 0u,
    DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING_FLAG_STANDARD = 1u << 0u,
};

enum GeometryFlags : unsigned int {
    GEOMETRY_FLAG_NONE = 0u,
    GEOMETRY_FLAG_DISABLE_ANYHIT = 1u << 0u,
    GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1u << 1u,
    GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 2u,
};

enum HitKind : unsigned int {
    HIT_KIND_TRIANGLE_FRONT_FACE = 0xfeu,
    HIT_KIND_TRIANGLE_BACK_FACE = 0xffu,
};

enum IndicesFormat : unsigned int {
    INDICES_FORMAT_NONE = 0u,
    INDICES_FORMAT_UNSIGNED_SHORT3 = 0x2102u,
    INDICES_FORMAT_UNSIGNED_INT3 = 0x2103u,
};

enum VertexFormat : unsigned int {
    VERTEX_FORMAT_NONE = 0u,
    VERTEX_FORMAT_FLOAT3 = 0x2121u,
    VERTEX_FORMAT_FLOAT2 = 0x2122u,
    VERTEX_FORMAT_HALF3 = 0x2123u,
    VERTEX_FORMAT_HALF2 = 0x2124u,
    VERTEX_FORMAT_SNORM16_3 = 0x2125u,
    VERTEX_FORMAT_SNORM16_2 = 0x2126u,
};

enum TransformFormat : unsigned int {
    TRANSFORM_FORMAT_NONE = 0u,
    TRANSFORM_FORMAT_MATRIX_FLOAT12 = 0x21e1u,
};

enum DisplacementMicromapBiasAndScaleFormat : unsigned int {
    DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_NONE = 0u,
    DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2 = 0x2241u,
    DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_HALF2 = 0x2242u,
};

enum DisplacementMicromapDirectionFormat : unsigned int {
    DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_NONE = 0u,
    DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3 = 0x2261u,
    DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3 = 0x2262u,
};

enum OpacityMicromapFormat : unsigned int {
    OPACITY_MICROMAP_FORMAT_NONE = 0u,
    OPACITY_MICROMAP_FORMAT_2_STATE = 1u,
    OPACITY_MICROMAP_FORMAT_4_STATE = 2u,
};

enum OpacityMicromapArrayIndexingMode : unsigned int {
    OPACITY_MICROMAP_ARRAY_INDEXING_MODE_NONE = 0u,
    OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR = 1u,
    OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED = 2u,
};

struct OpacityMicromapUsageCount {
    unsigned int count;
    unsigned int subdivisionLevel;
    OpacityMicromapFormat format;
};

struct BuildInputOpacityMicromap {
    OpacityMicromapArrayIndexingMode indexingMode;
    CUdeviceptr  opacityMicromapArray;
    CUdeviceptr  indexBuffer;
    unsigned int indexSizeInBytes;
    unsigned int indexStrideInBytes;
    unsigned int indexOffset;
    unsigned int numMicromapUsageCounts;
    const OpacityMicromapUsageCount* micromapUsageCounts;
};

struct RelocateInputOpacityMicromap {
    CUdeviceptr  opacityMicromapArray;
};

enum DisplacementMicromapFormat : unsigned int {
    DISPLACEMENT_MICROMAP_FORMAT_NONE = 0u,
    DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES = 1u,
    DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES = 2u,
    DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES = 3u,
};

enum DisplacementMicromapFlags : unsigned int {
    DISPLACEMENT_MICROMAP_FLAG_NONE = 0u,
    DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE = 1u << 0u,
    DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_BUILD = 1u << 1u,
};

enum DisplacementMicromapTriangleFlags : unsigned int {
    DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_NONE = 0u,
    DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_01 = 1u << 0u,
    DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_12 = 1u << 1u,
    DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_20 = 1u << 2u,
};

struct DisplacementMicromapDesc {
    unsigned int   byteOffset;
    unsigned short subdivisionLevel;
    unsigned short format;
};

struct DisplacementMicromapHistogramEntry {
    unsigned int                    count;
    unsigned int                    subdivisionLevel;
    DisplacementMicromapFormat format;
};

struct DisplacementMicromapArrayBuildInput {
    DisplacementMicromapFlags                 flags;
    CUdeviceptr                                    displacementValuesBuffer;
    CUdeviceptr                                    perDisplacementMicromapDescBuffer;
    unsigned int                                   perDisplacementMicromapDescStrideInBytes;
    unsigned int                                   numDisplacementMicromapHistogramEntries;
    const DisplacementMicromapHistogramEntry* displacementMicromapHistogramEntries;
};

struct DisplacementMicromapUsageCount {
    unsigned int                    count;
    unsigned int                    subdivisionLevel;
    DisplacementMicromapFormat format;
};

enum DisplacementMicromapArrayIndexingMode : unsigned int {
    DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_NONE = 0u,
    DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR = 1u,
    DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED = 2u,
};

struct BuildInputDisplacementMicromap {
    DisplacementMicromapArrayIndexingMode indexingMode;
    CUdeviceptr displacementMicromapArray;
    CUdeviceptr displacementMicromapIndexBuffer;
    CUdeviceptr vertexDirectionsBuffer;
    CUdeviceptr vertexBiasAndScaleBuffer;
    CUdeviceptr triangleFlagsBuffer;
    unsigned int displacementMicromapIndexOffset;
    unsigned int displacementMicromapIndexStrideInBytes;
    unsigned int displacementMicromapIndexSizeInBytes;
    DisplacementMicromapDirectionFormat vertexDirectionFormat;
    unsigned int                             vertexDirectionStrideInBytes;
    DisplacementMicromapBiasAndScaleFormat vertexBiasAndScaleFormat;
    unsigned int                                vertexBiasAndScaleStrideInBytes;
    unsigned int triangleFlagsStrideInBytes;
    unsigned int                               numDisplacementMicromapUsageCounts;
    const DisplacementMicromapUsageCount* displacementMicromapUsageCounts;
};

struct BuildInputTriangleArray {
    const CUdeviceptr* vertexBuffers;
    unsigned int numVertices;
    VertexFormat vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    IndicesFormat indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    TransformFormat transformFormat;
    BuildInputOpacityMicromap opacityMicromap;
    BuildInputDisplacementMicromap displacementMicromap;
};

struct RelocateInputTriangleArray {
    unsigned int numSbtRecords;
    RelocateInputOpacityMicromap opacityMicromap;
};

enum PrimitiveType : unsigned int {
    PRIMITIVE_TYPE_CUSTOM = 0x2500u,
    PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE = 0x2501u,
    PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE = 0x2502u,
    PRIMITIVE_TYPE_ROUND_LINEAR = 0x2503u,
    PRIMITIVE_TYPE_ROUND_CATMULLROM = 0x2504u,
    PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE = 0x2505u,
    PRIMITIVE_TYPE_SPHERE = 0x2506u,
    PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER = 0x2507u,
    PRIMITIVE_TYPE_TRIANGLE = 0x2531u,
    PRIMITIVE_TYPE_DISPLACED_MICROMESH_TRIANGLE = 0x2532u,
};

enum PrimitiveTypeFlags : unsigned int {
    PRIMITIVE_TYPE_FLAGS_CUSTOM = 1u << 0u,
    PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE = 1u << 1u,
    PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE = 1u << 2u,
    PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR = 1u << 3u,
    PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM = 1u << 4u,
    PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE = 1u << 5u,
    PRIMITIVE_TYPE_FLAGS_SPHERE = 1u << 6u,
    PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER = 1u << 7u,
    PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1u << 31u,
    PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE = 1u << 30u,
};

enum CurveEndcapFlags : unsigned int {
    CURVE_ENDCAP_DEFAULT = 0u,
    CURVE_ENDCAP_ON = 1u << 0u,
};

struct BuildInputCurveArray {
    PrimitiveType curveType;
    unsigned int numPrimitives;
    const CUdeviceptr* vertexBuffers;
    unsigned int numVertices;
    unsigned int vertexStrideInBytes;
    const CUdeviceptr* widthBuffers;
    unsigned int widthStrideInBytes;
    const CUdeviceptr* normalBuffers;
    unsigned int normalStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int indexStrideInBytes;
    unsigned int flag;
    unsigned int primitiveIndexOffset;
    unsigned int endcapFlags;
};

struct BuildInputSphereArray {
    const CUdeviceptr* vertexBuffers;
    unsigned int vertexStrideInBytes;
    unsigned int numVertices;
    const CUdeviceptr* radiusBuffers;
    unsigned int radiusStrideInBytes;
    int singleRadius;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
};

struct Aabb {
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
};

struct BuildInputCustomPrimitiveArray {
    const CUdeviceptr* aabbBuffers;
    unsigned int numPrimitives;
    unsigned int strideInBytes;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
};

struct BuildInputInstanceArray {
    CUdeviceptr instances;
    unsigned int numInstances;
    unsigned int instanceStride;
};

struct RelocateInputInstanceArray {
    unsigned int numInstances;
    CUdeviceptr traversableHandles;
};

enum BuildInputType : unsigned int {
    BUILD_INPUT_TYPE_TRIANGLES = 0x2141u,
    BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES = 0x2142u,
    BUILD_INPUT_TYPE_INSTANCES = 0x2143u,
    BUILD_INPUT_TYPE_INSTANCE_POINTERS = 0x2144u,
    BUILD_INPUT_TYPE_CURVES = 0x2145u,
    BUILD_INPUT_TYPE_SPHERES = 0x2146u,
};

struct BuildInput {
    BuildInputType type;
    union {
        BuildInputTriangleArray triangleArray;
        BuildInputCurveArray curveArray;
        BuildInputSphereArray sphereArray;
        BuildInputCustomPrimitiveArray customPrimitiveArray;
        BuildInputInstanceArray instanceArray;
        char pad[1024];
    };
};

struct RelocateInput {
    BuildInputType type;
    union {
        RelocateInputInstanceArray instanceArray;
        RelocateInputTriangleArray triangleArray;
    };
};

enum InstanceFlags : unsigned int {
    INSTANCE_FLAG_NONE = 0u,
    INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0u,
    INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1u,
    INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2u,
    INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3u,
    INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 1u << 4u,
    INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS = 1u << 5u,
};

struct Instance {
    float transform[12];
    unsigned int instanceId;
    unsigned int sbtOffset;
    unsigned int visibilityMask;
    unsigned int flags;
    TraversableHandle traversableHandle;
    unsigned int pad[2];
};

enum BuildFlags : unsigned int {
    BUILD_FLAG_NONE = 0u,
    BUILD_FLAG_ALLOW_UPDATE = 1u << 0u,
    BUILD_FLAG_ALLOW_COMPACTION = 1u << 1u,
    BUILD_FLAG_PREFER_FAST_TRACE = 1u << 2u,
    BUILD_FLAG_PREFER_FAST_BUILD = 1u << 3u,
    BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS = 1u << 4u,
    BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS = 1u << 5u,
    BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE = 1u << 6u,
    BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS = 1u << 7u,
};

enum OpacityMicromapFlags : unsigned int {
    OPACITY_MICROMAP_FLAG_NONE = 0u,
    OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE = 1u << 0u,
    OPACITY_MICROMAP_FLAG_PREFER_FAST_BUILD = 1u << 1u,
};

struct OpacityMicromapDesc {
    unsigned int  byteOffset;
    unsigned short subdivisionLevel;
    unsigned short format;
};

struct OpacityMicromapHistogramEntry {
    unsigned int               count;
    unsigned int               subdivisionLevel;
    OpacityMicromapFormat format;
};

struct OpacityMicromapArrayBuildInput {
    unsigned int flags;
    CUdeviceptr inputBuffer;
    CUdeviceptr perMicromapDescBuffer;
    unsigned int perMicromapDescStrideInBytes;
    unsigned int numMicromapHistogramEntries;
    const OpacityMicromapHistogramEntry* micromapHistogramEntries;
};

struct MicromapBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
};

struct MicromapBuffers {
    CUdeviceptr output;
    size_t outputSizeInBytes;
    CUdeviceptr temp;
    size_t tempSizeInBytes;
};

enum BuildOperation : unsigned int {
    BUILD_OPERATION_BUILD = 0x2161u,
    BUILD_OPERATION_UPDATE = 0x2162u,
};

enum MotionFlags : unsigned int {
    MOTION_FLAG_NONE = 0u,
    MOTION_FLAG_START_VANISH = 1u << 0u,
    MOTION_FLAG_END_VANISH = 1u << 1u,
};

struct MotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct AccelBuildOptions {
    unsigned int buildFlags;
    BuildOperation operation;
    MotionOptions motionOptions;
};

struct AccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

enum AccelPropertyType : unsigned int {
    PROPERTY_TYPE_COMPACTED_SIZE = 0x2181u,
    PROPERTY_TYPE_AABBS = 0x2182u,
};

struct AccelEmitDesc {
    CUdeviceptr result;
    AccelPropertyType type;
};

struct RelocationInfo {
    unsigned long long info[4];
};

struct StaticTransform {
    TraversableHandle child;
    unsigned int pad[2];
    float transform[12];
    float invTransform[12];
};

struct MatrixMotionTransform {
    TraversableHandle child;
    MotionOptions motionOptions;
    unsigned int pad[3];
    float transform[2][12];
};

struct SRTData {
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
};

struct SRTMotionTransform {
    TraversableHandle child;
    MotionOptions motionOptions;
    unsigned int pad[3];
    SRTData srtData[2];
};

enum TraversableType : unsigned int {
    TRAVERSABLE_TYPE_STATIC_TRANSFORM = 0x21c1u,
    TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM = 0x21c2u,
    TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM = 0x21c3u,
};

enum PixelFormat : unsigned int {
    PIXEL_FORMAT_HALF1 = 0x220au,
    PIXEL_FORMAT_HALF2 = 0x2207u,
    PIXEL_FORMAT_HALF3 = 0x2201u,
    PIXEL_FORMAT_HALF4 = 0x2202u,
    PIXEL_FORMAT_FLOAT1 = 0x220bu,
    PIXEL_FORMAT_FLOAT2 = 0x2208u,
    PIXEL_FORMAT_FLOAT3 = 0x2203u,
    PIXEL_FORMAT_FLOAT4 = 0x2204u,
    PIXEL_FORMAT_UCHAR3 = 0x2205u,
    PIXEL_FORMAT_UCHAR4 = 0x2206u,
    PIXEL_FORMAT_INTERNAL_GUIDE_LAYER = 0x2209u,
};

struct Image2D {
    CUdeviceptr data;
    unsigned int width;
    unsigned int height;
    unsigned int rowStrideInBytes;
    unsigned int pixelStrideInBytes;
    PixelFormat format;
};

enum DenoiserModelKind : unsigned int {
    DENOISER_MODEL_KIND_LDR = 0x2322u,
    DENOISER_MODEL_KIND_HDR = 0x2323u,
    DENOISER_MODEL_KIND_AOV = 0x2324u,
    DENOISER_MODEL_KIND_TEMPORAL = 0x2325u,
    DENOISER_MODEL_KIND_TEMPORAL_AOV = 0x2326u,
    DENOISER_MODEL_KIND_UPSCALE2X = 0x2327u,
    DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X = 0x2328u,
};

enum DenoiserAlphaMode : unsigned int {
    DENOISER_ALPHA_MODE_COPY = 0u,
    DENOISER_ALPHA_MODE_DENOISE = 1u,
};

struct DenoiserOptions {
    unsigned int guideAlbedo;
    unsigned int guideNormal;
    DenoiserAlphaMode denoiseAlpha;
};

struct DenoiserGuideLayer {
    Image2D  albedo;
    Image2D  normal;
    Image2D  flow;
    Image2D  previousOutputInternalGuideLayer;
    Image2D  outputInternalGuideLayer;
    Image2D flowTrustworthiness;
};

enum DenoiserAOVType : unsigned int {
    DENOISER_AOV_TYPE_NONE = 0u,
    DENOISER_AOV_TYPE_BEAUTY = 0x7000u,
    DENOISER_AOV_TYPE_SPECULAR = 0x7001u,
    DENOISER_AOV_TYPE_REFLECTION = 0x7002u,
    DENOISER_AOV_TYPE_REFRACTION = 0x7003u,
    DENOISER_AOV_TYPE_DIFFUSE = 0x7004u,
};

struct DenoiserLayer {
    Image2D  input;
    Image2D  previousOutput;
    Image2D  output;
    DenoiserAOVType type;
};

struct DenoiserParams {
    CUdeviceptr  hdrIntensity;
    float        blendFactor;
    CUdeviceptr  hdrAverageColor;
    unsigned int temporalModeUsePreviousLayers;
};

struct DenoiserSizes {
    size_t stateSizeInBytes;
    size_t withOverlapScratchSizeInBytes;
    size_t withoutOverlapScratchSizeInBytes;
    unsigned int overlapWindowSizeInPixels;
    size_t computeAverageColorSizeInBytes;
    size_t computeIntensitySizeInBytes;
    size_t internalGuideLayerPixelSizeInBytes;
};

enum RayFlags : unsigned int {
    RAY_FLAG_NONE = 0u,
    RAY_FLAG_DISABLE_ANYHIT = 1u << 0u,
    RAY_FLAG_ENFORCE_ANYHIT = 1u << 1u,
    RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1u << 2u,
    RAY_FLAG_DISABLE_CLOSESTHIT = 1u << 3u,
    RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1u << 4u,
    RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1u << 5u,
    RAY_FLAG_CULL_DISABLED_ANYHIT = 1u << 6u,
    RAY_FLAG_CULL_ENFORCED_ANYHIT = 1u << 7u,
    RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 1u << 10u,
};

enum TransformType : unsigned int {
    TRANSFORM_TYPE_NONE = 0u,
    TRANSFORM_TYPE_STATIC_TRANSFORM = 1u,
    TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM = 2u,
    TRANSFORM_TYPE_SRT_MOTION_TRANSFORM = 3u,
    TRANSFORM_TYPE_INSTANCE = 4u,
};

enum TraversableGraphFlags : unsigned int {
    TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0u,
    TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0u,
    TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1u,
};

enum CompileOptimizationLevel : unsigned int {
    COMPILE_OPTIMIZATION_DEFAULT = 0u,
    COMPILE_OPTIMIZATION_LEVEL_0 = 0x2340u,
    COMPILE_OPTIMIZATION_LEVEL_1 = 0x2341u,
    COMPILE_OPTIMIZATION_LEVEL_2 = 0x2342u,
    COMPILE_OPTIMIZATION_LEVEL_3 = 0x2343u,
};

enum CompileDebugLevel : unsigned int {
    COMPILE_DEBUG_LEVEL_DEFAULT = 0u,
    COMPILE_DEBUG_LEVEL_NONE = 0x2350u,
    COMPILE_DEBUG_LEVEL_MINIMAL = 0x2351u,
    COMPILE_DEBUG_LEVEL_MODERATE = 0x2353u,
    COMPILE_DEBUG_LEVEL_FULL = 0x2352u,
};

enum ModuleCompileState : unsigned int {
    MODULE_COMPILE_STATE_NOT_STARTED = 0x2360u,
    MODULE_COMPILE_STATE_STARTED = 0x2361u,
    MODULE_COMPILE_STATE_IMPENDING_FAILURE = 0x2362u,
    MODULE_COMPILE_STATE_FAILED = 0x2363u,
    MODULE_COMPILE_STATE_COMPLETED = 0x2364u,
};

struct ModuleCompileBoundValueEntry {
    size_t pipelineParamOffsetInBytes;
    size_t sizeInBytes;
    const void* boundValuePtr;
    const char* annotation;
};

enum PayloadTypeID : unsigned int {
    PAYLOAD_TYPE_DEFAULT = 0u,
    PAYLOAD_TYPE_ID_0 = 1u << 0u,
    PAYLOAD_TYPE_ID_1 = 1u << 1u,
    PAYLOAD_TYPE_ID_2 = 1u << 2u,
    PAYLOAD_TYPE_ID_3 = 1u << 3u,
    PAYLOAD_TYPE_ID_4 = 1u << 4u,
    PAYLOAD_TYPE_ID_5 = 1u << 5u,
    PAYLOAD_TYPE_ID_6 = 1u << 6u,
    PAYLOAD_TYPE_ID_7 = 1u << 7u,
};

enum PayloadSemantics : unsigned int {
    PAYLOAD_SEMANTICS_TRACE_CALLER_NONE = 0u,
    PAYLOAD_SEMANTICS_TRACE_CALLER_READ = 1u << 0u,
    PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE = 2u << 0u,
    PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE = 3u << 0u,
    PAYLOAD_SEMANTICS_CH_NONE = 0u,
    PAYLOAD_SEMANTICS_CH_READ = 1u << 2u,
    PAYLOAD_SEMANTICS_CH_WRITE = 2u << 2u,
    PAYLOAD_SEMANTICS_CH_READ_WRITE = 3u << 2u,
    PAYLOAD_SEMANTICS_MS_NONE = 0u,
    PAYLOAD_SEMANTICS_MS_READ = 1u << 4u,
    PAYLOAD_SEMANTICS_MS_WRITE = 2u << 4u,
    PAYLOAD_SEMANTICS_MS_READ_WRITE = 3u << 4u,
    PAYLOAD_SEMANTICS_AH_NONE = 0u,
    PAYLOAD_SEMANTICS_AH_READ = 1u << 6u,
    PAYLOAD_SEMANTICS_AH_WRITE = 2u << 6u,
    PAYLOAD_SEMANTICS_AH_READ_WRITE = 3u << 6u,
    PAYLOAD_SEMANTICS_IS_NONE = 0u,
    PAYLOAD_SEMANTICS_IS_READ = 1u << 8u,
    PAYLOAD_SEMANTICS_IS_WRITE = 2u << 8u,
    PAYLOAD_SEMANTICS_IS_READ_WRITE = 3u << 8u,
};

struct PayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct ModuleCompileOptions {
    int maxRegisterCount;
    CompileOptimizationLevel optLevel;
    CompileDebugLevel debugLevel;
    const ModuleCompileBoundValueEntry* boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    const PayloadType* payloadTypes;
};

enum ProgramGroupKind : unsigned int {
    PROGRAM_GROUP_KIND_RAYGEN = 0x2421u,
    PROGRAM_GROUP_KIND_MISS = 0x2422u,
    PROGRAM_GROUP_KIND_EXCEPTION = 0x2423u,
    PROGRAM_GROUP_KIND_HITGROUP = 0x2424u,
    PROGRAM_GROUP_KIND_CALLABLES = 0x2425u,
};

enum ProgramGroupFlags : unsigned int {
    PROGRAM_GROUP_FLAGS_NONE = 0u,
};

struct ProgramGroupSingleModule {
    Module module;
    const char* entryFunctionName;
};

struct ProgramGroupHitgroup {
    Module moduleCH;
    const char* entryFunctionNameCH;
    Module moduleAH;
    const char* entryFunctionNameAH;
    Module moduleIS;
    const char* entryFunctionNameIS;
};

struct ProgramGroupCallables {
    Module moduleDC;
    const char* entryFunctionNameDC;
    Module moduleCC;
    const char* entryFunctionNameCC;
};

struct ProgramGroupDesc {
    ProgramGroupKind kind;
    unsigned int flags;
    union {
        ProgramGroupSingleModule raygen;
        ProgramGroupSingleModule miss;
        ProgramGroupSingleModule exception;
        ProgramGroupCallables callables;
        ProgramGroupHitgroup hitgroup;
    };
};

struct ProgramGroupOptions {
    const PayloadType* payloadType;
};

enum ExceptionCodes : int {
    EXCEPTION_CODE_STACK_OVERFLOW = -1,
    EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED = -2,
};

enum ExceptionFlags : unsigned int {
    EXCEPTION_FLAG_NONE = 0u,
    EXCEPTION_FLAG_STACK_OVERFLOW = 1u << 0u,
    EXCEPTION_FLAG_TRACE_DEPTH = 1u << 1u,
    EXCEPTION_FLAG_USER = 1u << 2u,
};

struct PipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
    int allowOpacityMicromaps;
};

struct PipelineLinkOptions {
    unsigned int maxTraceDepth;
};

struct ShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

struct StackSizes {
    unsigned int cssRG;
    unsigned int cssMS;
    unsigned int cssCH;
    unsigned int cssAH;
    unsigned int cssIS;
    unsigned int cssCC;
    unsigned int dssDC;
};

enum QueryFunctionTableOptions : unsigned int {
    QUERY_FUNCTION_TABLE_OPTION_DUMMY = 0u,
};

struct BuiltinISOptions {
    PrimitiveType        builtinISModuleType;
    int                       usesMotionBlur;
    unsigned int              buildFlags;
    unsigned int              curveEndcapFlags;
};

struct InvalidRayExceptionDetails {
    std::array<float, 3> origin;
    std::array<float, 3> direction;
    float  tmin;
    float  tmax;
    float  time;
};

struct ParameterMismatchExceptionDetails {
    unsigned int expectedParameterCount;
    unsigned int passedArgumentCount;
    unsigned int sbtIndex;
    char*        callableName;
};

// function table
struct FunctionTable {

    const char* (*getErrorName)(Result result);

    const char* (*getErrorString)(Result result);

    Result (*deviceContextCreate)(CUcontext fromContext,
                                  const DeviceContextOptions* options,
                                  DeviceContext* context);

    Result (*deviceContextDestroy)(DeviceContext context);

    Result (*deviceContextGetProperty)(DeviceContext context,
                                       DeviceProperty property,
                                       void* value,
                                       size_t sizeInBytes);

    Result (*deviceContextSetLogCallback)(DeviceContext context,
                                          LogCallback callbackFunction,
                                          void* callbackData,
                                          unsigned int callbackLevel);

    Result (*deviceContextSetCacheEnabled)(DeviceContext context,
                                           int enabled);

    Result (*deviceContextSetCacheLocation)(DeviceContext context,
                                            const char* location);

    Result (*deviceContextSetCacheDatabaseSizes)(DeviceContext context,
                                                 size_t lowWaterMark,
                                                 size_t highWaterMark);

    Result (*deviceContextGetCacheEnabled)(DeviceContext context,
                                           int* enabled);

    Result (*deviceContextGetCacheLocation)(DeviceContext context,
                                            char* location,
                                            size_t locationSize);

    Result (*deviceContextGetCacheDatabaseSizes)(DeviceContext context,
                                                 size_t* lowWaterMark,
                                                 size_t* highWaterMark);

    Result (*moduleCreate)(DeviceContext context,
                           const ModuleCompileOptions* moduleCompileOptions,
                           const PipelineCompileOptions* pipelineCompileOptions,
                           const char* input,
                           size_t inputSize,
                           char* logString,
                           size_t* logStringSize,
                           Module* module);

    Result (*moduleCreateWithTasks)(DeviceContext context,
                                    const ModuleCompileOptions* moduleCompileOptions,
                                    const PipelineCompileOptions* pipelineCompileOptions,
                                    const char* input,
                                    size_t inputSize,
                                    char* logString,
                                    size_t* logStringSize,
                                    Module* module,
                                    Task* firstTask);

    Result (*moduleGetCompilationState)(Module module,
                                        ModuleCompileState* state);

    Result (*moduleDestroy)(Module module);

    Result(*builtinISModuleGet)(DeviceContext context,
                                 const ModuleCompileOptions* moduleCompileOptions,
                                 const PipelineCompileOptions* pipelineCompileOptions,
                                 const BuiltinISOptions* builtinISOptions,
                                 Module* builtinModule);

    Result (*taskExecute)(Task task,
                          Task* additionalTasks,
                          unsigned int maxNumAdditionalTasks,
                          unsigned int* numAdditionalTasksCreated);

    Result (*programGroupCreate)(DeviceContext context,
                                 const ProgramGroupDesc* programDescriptions,
                                 unsigned int numProgramGroups,
                                 const ProgramGroupOptions* options,
                                 char* logString,
                                 size_t* logStringSize,
                                 ProgramGroup* programGroups);

    Result (*programGroupDestroy)(ProgramGroup programGroup);

    Result (*programGroupGetStackSize)(ProgramGroup programGroup,
                                       StackSizes* stackSizes,
                                       Pipeline pipeline);

    Result (*pipelineCreate)(DeviceContext context,
                             const PipelineCompileOptions* pipelineCompileOptions,
                             const PipelineLinkOptions* pipelineLinkOptions,
                             const ProgramGroup* programGroups,
                             unsigned int numProgramGroups,
                             char* logString,
                             size_t* logStringSize,
                             Pipeline* pipeline);

    Result (*pipelineDestroy)(Pipeline pipeline);

    Result (*pipelineSetStackSize)(Pipeline pipeline,
                                   unsigned int directCallableStackSizeFromTraversal,
                                   unsigned int directCallableStackSizeFromState,
                                   unsigned int continuationStackSize,
                                   unsigned int maxTraversableGraphDepth);

    Result (*accelComputeMemoryUsage)(DeviceContext context,
                                      const AccelBuildOptions* accelOptions,
                                      const BuildInput* buildInputs,
                                      unsigned int numBuildInputs,
                                      AccelBufferSizes* bufferSizes);

    Result (*accelBuild)(DeviceContext context,
                         CUstream stream,
                         const AccelBuildOptions* accelOptions,
                         const BuildInput* buildInputs,
                         unsigned int numBuildInputs,
                         CUdeviceptr tempBuffer,
                         size_t tempBufferSizeInBytes,
                         CUdeviceptr outputBuffer,
                         size_t outputBufferSizeInBytes,
                         TraversableHandle* outputHandle,
                         const AccelEmitDesc* emittedProperties,
                         unsigned int numEmittedProperties);

    Result (*accelGetRelocationInfo)(DeviceContext context,
                                     TraversableHandle handle,
                                     RelocationInfo* info);

    Result (*checkRelocationCompatibility)(DeviceContext context,
                                           const RelocationInfo* info,
                                           int* compatible);

    Result (*accelRelocate)(DeviceContext context,
                            CUstream stream,
                            const RelocationInfo* info,
                            const RelocateInput* relocateInputs,
                            size_t numRelocateInputs,
                            CUdeviceptr targetAccel,
                            size_t targetAccelSizeInBytes,
                            TraversableHandle* targetHandle);

    Result (*accelCompact)(DeviceContext context,
                           CUstream stream,
                           TraversableHandle inputHandle,
                           CUdeviceptr outputBuffer,
                           size_t outputBufferSizeInBytes,
                           TraversableHandle* outputHandle);

    Result (*accelEmitProperty)(DeviceContext context,
                                CUstream stream,
                                TraversableHandle handle,
                                const AccelEmitDesc* emittedProperty);

    Result (*convertPointerToTraversableHandle)(DeviceContext onDevice,
                                                CUdeviceptr pointer,
                                                TraversableType traversableType,
                                                TraversableHandle* traversableHandle);

    Result (*opacityMicromapArrayComputeMemoryUsage)(DeviceContext context,
                                                     const OpacityMicromapArrayBuildInput* buildInput,
                                                     MicromapBufferSizes* bufferSizes);

    Result (*opacityMicromapArrayBuild)(DeviceContext context,
                                        CUstream stream,
                                        const OpacityMicromapArrayBuildInput* buildInput,
                                        const MicromapBuffers* buffers);

    Result (*opacityMicromapArrayGetRelocationInfo)(DeviceContext context,
                                                    CUdeviceptr opacityMicromapArray,
                                                    RelocationInfo* info);

    Result (*opacityMicromapArrayRelocate)(DeviceContext context,
                                           CUstream stream,
                                           const RelocationInfo* info,
                                           CUdeviceptr targetOpacityMicromapArray,
                                           size_t targetOpacityMicromapArraySizeInBytes);

    Result (*displacementMicromapArrayComputeMemoryUsage)(DeviceContext context,
                                                          const DisplacementMicromapArrayBuildInput* buildInput,
                                                          MicromapBufferSizes* bufferSizes);

    Result (*displacementMicromapArrayBuild)(DeviceContext context,
                                             CUstream stream,
                                             const DisplacementMicromapArrayBuildInput* buildInput,
                                             const MicromapBuffers* buffers);

    Result (*sbtRecordPackHeader)(ProgramGroup programGroup,
                                  void* sbtRecordHeaderHostPointer);

    Result (*launch)(Pipeline pipeline,
                     CUstream stream,
                     CUdeviceptr pipelineParams,
                     size_t pipelineParamsSize,
                     const ShaderBindingTable* sbt,
                     unsigned int width,
                     unsigned int height,
                     unsigned int depth);

    Result (*denoiserCreate)(DeviceContext context,
                             DenoiserModelKind modelKind,
                             const DenoiserOptions* options,
                             Denoiser* returnHandle);

    Result (*denoiserDestroy)(Denoiser handle);

    Result (*denoiserComputeMemoryResources)(const Denoiser handle,
                                             unsigned int maximumInputWidth,
                                             unsigned int maximumInputHeight,
                                             DenoiserSizes* returnSizes);

    Result (*denoiserSetup)(Denoiser denoiser,
                            CUstream stream,
                            unsigned int inputWidth,
                            unsigned int inputHeight,
                            CUdeviceptr state,
                            size_t stateSizeInBytes,
                            CUdeviceptr scratch,
                            size_t scratchSizeInBytes);

    Result (*denoiserInvoke)(Denoiser denoiser,
                             CUstream stream,
                             const DenoiserParams* params,
                             CUdeviceptr denoiserState,
                             size_t denoiserStateSizeInBytes,
                             const DenoiserGuideLayer * guideLayer,
                             const DenoiserLayer * layers,
                             unsigned int numLayers,
                             unsigned int inputOffsetX,
                             unsigned int inputOffsetY,
                             CUdeviceptr scratch,
                             size_t scratchSizeInBytes);

    Result (*denoiserComputeIntensity)(Denoiser handle,
                                       CUstream stream,
                                       const Image2D* inputImage,
                                       CUdeviceptr outputIntensity,
                                       CUdeviceptr scratch,
                                       size_t scratchSizeInBytes);

    Result (*denoiserComputeAverageColor)(Denoiser handle,
                                          CUstream stream,
                                          const Image2D* inputImage,
                                          CUdeviceptr outputAverageColor,
                                          CUdeviceptr scratch,
                                          size_t scratchSizeInBytes);

    Result (*denoiserCreateWithUserModel)(DeviceContext context,
                                          const void * data,
                                          size_t dataSizeInBytes,
                                          Denoiser* returnHandle);
};

// API
[[nodiscard]] const FunctionTable &api() noexcept;

} // namespace luisa::compute::optix
