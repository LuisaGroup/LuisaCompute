// Standalone HIPRT test - bypasses LuisaCompute runtime entirely
// Tests HIPRT scene traversal directly on the current GPU
// to isolate whether HIPRT SDK works on gfx1200 (RDNA 4)

#include "ut/ut.hpp"
#include "test_device.h"

#if __has_include(<hip/hip_runtime.h>) && __has_include(<hip/hiprtc.h>) && __has_include(<hiprt/hiprt.h>)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hiprt/hiprt.h>

using namespace boost::ut;
using namespace boost::ut::literals;

#define CHECK_HIP(call)                                               \
    do {                                                              \
        hipError_t err = (call);                                      \
        if (err != hipSuccess) {                                      \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n",           \
                    err, hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                                  \
        }                                                             \
    } while (0)

#define CHECK_HIPRT(call)                                \
    do {                                                 \
        hiprtError err = (call);                         \
        if (err != hiprtSuccess) {                       \
            fprintf(stderr, "HIPRT error %d at %s:%d\n", \
                    (int)err, __FILE__, __LINE__);       \
            exit(1);                                     \
        }                                                \
    } while (0)

#define CHECK_HIPRTC(call)                                                    \
    do {                                                                      \
        hiprtcResult err = (call);                                            \
        if (err != HIPRTC_SUCCESS) {                                          \
            fprintf(stderr, "HIPRTC error %d (%s) at %s:%d\n",                \
                    (int)err, hiprtcGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Kernel source: simple scene intersection test
// Takes hiprtScene directly as pointer (like tutorials)
static const char *kernel_source = R"(
#include <hiprt/hiprt_device.h>

extern "C" __global__ void TestGeomKernel(
    hiprtGeometry geom,
    float *out_t,
    int *out_hit,
    int num_rays)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    // First thread prints diagnostic info
    if (idx == 0) {
#ifdef HIPRT_RTIP
        printf("HIPRT_RTIP = %d\n", HIPRT_RTIP);
#else
        printf("HIPRT_RTIP is NOT defined\n");
#endif
#ifdef __KERNELCC__
        printf("__KERNELCC__ is defined\n");
#else
        printf("__KERNELCC__ is NOT defined\n");
#endif
#ifdef HIP_VERSION_MAJOR
        printf("HIP_VERSION_MAJOR = %d\n", HIP_VERSION_MAJOR);
#else
        printf("HIP_VERSION_MAJOR is NOT defined\n");
#endif
    }

    hiprtRay ray;
    ray.origin = hiprtFloat3{-0.25f + (float)idx * 0.05f, 0.0f, -1.0f};
    ray.direction = hiprtFloat3{0.0f, 0.0f, 1.0f};

    hiprtGeomTraversalClosest traversal(geom, ray);
    hiprtHit hit = traversal.getNextHit();

    out_t[idx] = hit.hasHit() ? hit.t : -1.0f;
    out_hit[idx] = hit.hasHit() ? 1 : 0;
}

extern "C" __global__ void TestSceneKernel(
    hiprtScene scene,
    float *out_t,
    int *out_hit,
    int num_rays)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rays) return;

    hiprtRay ray;
    ray.origin = hiprtFloat3{-0.25f + (float)idx * 0.05f, 0.0f, -1.0f};
    ray.direction = hiprtFloat3{0.0f, 0.0f, 1.0f};

    hiprtSceneTraversalClosest traversal(scene, ray, 0xFFFFFFFF);
    hiprtHit hit = traversal.getNextHit();

    out_t[idx] = hit.hasHit() ? hit.t : -1.0f;
    out_hit[idx] = hit.hasHit() ? 1 : 0;
}
)";

void test_hiprt_standalone(luisa::compute::Device &device) {
    (void)device;

    printf("=== Standalone HIPRT Test ===\n");

    // Init HIP
    CHECK_HIP(hipInit(0));
    hipDevice_t device;
    CHECK_HIP(hipDeviceGet(&device, 0));

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    printf("GCN Arch Name: %s\n", props.gcnArchName);

    // Create HIPRT context
    hiprtContextCreationInput ctxtInput;
    ctxtInput.ctxt = nullptr;// use default HIP context
    ctxtInput.device = device;
    ctxtInput.deviceType = hiprtDeviceAMD;

    hiprtContext hiprt_ctx;
    CHECK_HIPRT(hiprtCreateContext(HIPRT_API_VERSION, ctxtInput, hiprt_ctx));
    printf("HIPRT context created successfully.\n");

    // ===== Build a simple triangle mesh =====
    // Using EXACT same layout as tutorial 02_scene_intersection
    constexpr float S = 0.5f;
    constexpr float T = 0.8f;
    // Vertices at Z=0, range ~[-0.5, 0.5] in X/Y
    float vertices[] = {
        S,
        S,
        0.0f,// v0
        S + T * S,
        -S * S,
        0.0f,// v1
        S - T * S,
        -S * S,
        0.0f,// v2
        -S,
        S,
        0.0f,// v3
        -S + T * S,
        -S * S,
        0.0f,// v4
        -S - T * S,
        -S * S,
        0.0f,// v5
    };
    uint32_t indices[] = {0, 1, 2, 3, 4, 5};

    hiprtTriangleMeshPrimitive mesh;
    mesh.triangleCount = 2;
    mesh.triangleStride = sizeof(uint32_t) * 3;
    mesh.vertexCount = 6;
    mesh.vertexStride = sizeof(float) * 3;

    CHECK_HIP(hipMalloc(&mesh.triangleIndices, sizeof(indices)));
    CHECK_HIP(hipMemcpy(mesh.triangleIndices, indices, sizeof(indices), hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc(&mesh.vertices, sizeof(vertices)));
    CHECK_HIP(hipMemcpy(mesh.vertices, vertices, sizeof(vertices), hipMemcpyHostToDevice));

    hiprtGeometryBuildInput geomInput;
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh;

    hiprtBuildOptions options;
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t geomTempSize;
    CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(hiprt_ctx, geomInput, options, geomTempSize));
    hiprtDevicePtr geomTemp;
    CHECK_HIP(hipMalloc(&geomTemp, geomTempSize));

    hiprtGeometry geom;
    CHECK_HIPRT(hiprtCreateGeometry(hiprt_ctx, geomInput, options, geom));
    CHECK_HIPRT(hiprtBuildGeometry(hiprt_ctx, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom));
    printf("Geometry built successfully.\n");

    // ===== Build scene with one instance =====
    hiprtInstance instance;
    instance.type = hiprtInstanceTypeGeometry;
    instance.geometry = geom;

    hiprtSceneBuildInput sceneInput;
    sceneInput.instanceCount = 1;
    sceneInput.instanceMasks = nullptr;
    sceneInput.instanceTransformHeaders = nullptr;

    CHECK_HIP(hipMalloc(&sceneInput.instances, sizeof(hiprtInstance)));
    CHECK_HIP(hipMemcpy(sceneInput.instances, &instance, sizeof(hiprtInstance), hipMemcpyHostToDevice));

    // Use SRT frame (identity transform) — same as tutorials
    hiprtFrameSRT frame;
    frame.translation = {0.0f, 0.0f, 0.0f};
    frame.scale = {1.0f, 1.0f, 1.0f};
    frame.rotation = {0.0f, 0.0f, 1.0f, 0.0f};// quaternion: identity = (0,0,sin(0),cos(0)) but HIPRT uses (x,y,z,w)
    sceneInput.frameCount = 1;
    CHECK_HIP(hipMalloc(&sceneInput.instanceFrames, sizeof(hiprtFrameSRT)));
    CHECK_HIP(hipMemcpy(sceneInput.instanceFrames, &frame, sizeof(hiprtFrameSRT), hipMemcpyHostToDevice));

    size_t sceneTempSize;
    CHECK_HIPRT(hiprtGetSceneBuildTemporaryBufferSize(hiprt_ctx, sceneInput, options, sceneTempSize));
    hiprtDevicePtr sceneTemp;
    CHECK_HIP(hipMalloc(&sceneTemp, sceneTempSize));

    hiprtScene scene = nullptr;
    CHECK_HIPRT(hiprtCreateScene(hiprt_ctx, sceneInput, options, scene));
    CHECK_HIPRT(hiprtBuildScene(hiprt_ctx, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene));
    CHECK_HIP(hipDeviceSynchronize());
    printf("Scene built successfully. scene=%p\n", (void *)scene);

    // ===== Compile kernel using hiprtc =====
    hiprtcProgram prog;
    CHECK_HIPRTC(hiprtcCreateProgram(&prog, kernel_source, "test_kernel.hip", 0, nullptr, nullptr));
    CHECK_HIPRTC(hiprtcAddNameExpression(prog, "TestGeomKernel"));
    CHECK_HIPRTC(hiprtcAddNameExpression(prog, "TestSceneKernel"));

    std::string hiprt_include = "-I" + std::string(HIPRT_SDK_INCLUDE_DIR);

    // Let HIPRT_RTIP be determined by target macros (gfx1201 -> RTIP=31 with ROCm 7+)
    const char *compile_options[] = {
        "-fgpu-rdc",
        "-Xclang",
        "-disable-llvm-passes",
        "-Xclang",
        "-mno-constructor-aliases",
        "-std=c++17",
        hiprt_include.c_str(),
    };
    int num_options = sizeof(compile_options) / sizeof(compile_options[0]);

    hiprtcResult compileResult = hiprtcCompileProgram(prog, num_options, compile_options);
    if (compileResult != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, log.data());
        fprintf(stderr, "Compilation failed:\n%s\n", log.c_str());
        exit(1);
    }
    printf("Kernel compiled successfully.\n");

    // Get bitcode
    size_t bcSize;
    CHECK_HIPRTC(hiprtcGetBitcodeSize(prog, &bcSize));
    std::vector<char> bitcode(bcSize);
    CHECK_HIPRTC(hiprtcGetBitcode(prog, bitcode.data()));
    printf("Got bitcode: %zu bytes\n", bcSize);

    // ===== Build trace kernels =====
    const char *funcNames[] = {"TestGeomKernel", "TestSceneKernel"};
    hiprtApiFunction apiFuncs[2] = {nullptr, nullptr};
    CHECK_HIPRT(hiprtBuildTraceKernelsFromBitcode(
        hiprt_ctx,
        2,
        funcNames,
        "test_module",
        bitcode.data(),
        bcSize,
        0,// numGeomTypes
        1,// numRayTypes
        nullptr,
        apiFuncs,
        false));

    hipFunction_t geomFunc = reinterpret_cast<hipFunction_t>(apiFuncs[0]);
    hipFunction_t sceneFunc = reinterpret_cast<hipFunction_t>(apiFuncs[1]);
    printf("Trace kernels built. geomFunc=%p, sceneFunc=%p\n", (void *)geomFunc, (void *)sceneFunc);

    // ===== Allocate output buffers =====
    constexpr int NUM_RAYS = 16;
    float *d_out_t;
    int *d_out_hit;
    CHECK_HIP(hipMalloc(&d_out_t, NUM_RAYS * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out_hit, NUM_RAYS * sizeof(int)));

    float h_out_t[NUM_RAYS];
    int h_out_hit[NUM_RAYS];

    // ===== Test 1: Geometry intersection (no scene, just BVH) =====
    printf("\n--- Test 1: Geometry Intersection ---\n");
    CHECK_HIP(hipMemset(d_out_t, 0, NUM_RAYS * sizeof(float)));
    CHECK_HIP(hipMemset(d_out_hit, 0, NUM_RAYS * sizeof(int)));

    {
        int num_rays = NUM_RAYS;
        void *args[] = {&geom, &d_out_t, &d_out_hit, &num_rays};
        printf("Launching geom kernel: geom=%p\n", (void *)geom);

        CHECK_HIP(hipModuleLaunchKernel(
            geomFunc,
            1, 1, 1,
            NUM_RAYS, 1, 1,
            0, 0,
            args, nullptr));
        CHECK_HIP(hipDeviceSynchronize());
        printf("Geom kernel executed OK.\n");

        CHECK_HIP(hipMemcpy(h_out_t, d_out_t, NUM_RAYS * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_out_hit, d_out_hit, NUM_RAYS * sizeof(int), hipMemcpyDeviceToHost));

        int hits = 0;
        for (int i = 0; i < NUM_RAYS; i++) {
            printf("  Ray %d: hit=%d t=%.3f (origin=%.1f,0,-1)\n", i, h_out_hit[i], h_out_t[i], i * 0.1f);
            hits += h_out_hit[i];
        }
        printf("Geom hits: %d / %d\n", hits, NUM_RAYS);
    }

    // ===== Test 2: Scene intersection =====
    printf("\n--- Test 2: Scene Intersection ---\n");
    CHECK_HIP(hipMemset(d_out_t, 0, NUM_RAYS * sizeof(float)));
    CHECK_HIP(hipMemset(d_out_hit, 0, NUM_RAYS * sizeof(int)));

    {
        int num_rays = NUM_RAYS;
        void *args[] = {&scene, &d_out_t, &d_out_hit, &num_rays};
        printf("Launching scene kernel: scene=%p\n", (void *)scene);

        CHECK_HIP(hipModuleLaunchKernel(
            sceneFunc,
            1, 1, 1,
            NUM_RAYS, 1, 1,
            0, 0,
            args, nullptr));
        CHECK_HIP(hipDeviceSynchronize());
        printf("Scene kernel executed OK.\n");

        CHECK_HIP(hipMemcpy(h_out_t, d_out_t, NUM_RAYS * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_out_hit, d_out_hit, NUM_RAYS * sizeof(int), hipMemcpyDeviceToHost));

        int hits = 0;
        for (int i = 0; i < NUM_RAYS; i++) {
            printf("  Ray %d: hit=%d t=%.3f (origin=%.1f,0,-1)\n", i, h_out_hit[i], h_out_t[i], i * 0.1f);
            hits += h_out_hit[i];
        }
        printf("Scene hits: %d / %d\n", hits, NUM_RAYS);

        if (hits > 0) {
            printf("\n*** HIPRT SCENE TRAVERSAL WORKS ON THIS GPU! ***\n");
        } else {
            printf("\n*** WARNING: No scene hits detected. ***\n");
        }
    }

    // Cleanup
    CHECK_HIP(hipFree(d_out_t));
    CHECK_HIP(hipFree(d_out_hit));
    CHECK_HIP(hipFree(geomTemp));
    CHECK_HIP(hipFree(sceneTemp));
    CHECK_HIP(hipFree(sceneInput.instances));
    CHECK_HIP(hipFree(sceneInput.instanceFrames));
    CHECK_HIP(hipFree(mesh.triangleIndices));
    CHECK_HIP(hipFree(mesh.vertices));

    CHECK_HIPRT(hiprtDestroyGeometry(hiprt_ctx, geom));
    CHECK_HIPRT(hiprtDestroyScene(hiprt_ctx, scene));
    CHECK_HIPRT(hiprtDestroyContext(hiprt_ctx));

    hiprtcDestroyProgram(&prog);

    printf("=== Test Complete ===\n");
    return;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_hiprt_standalone(device);
}
#else

using namespace boost::ut;
using namespace boost::ut::literals;

void test_hiprt_standalone(luisa::compute::Device &device) {
    (void)device;
    return;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_hiprt_standalone(device);
}

#endif
