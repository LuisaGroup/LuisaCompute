#include <hip/hip_runtime.h>

namespace {

struct PackedPosition {
    float x;
    float y;
    float z;
};

struct PackedTriangle {
    unsigned int i0;
    unsigned int i1;
    unsigned int i2;
};

struct MotionTriangleAABB {
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
};

struct alignas(16) MotionMeshValidationStatus {
    unsigned long long failure_key;
    unsigned long long reserved;
};

static_assert(sizeof(PackedPosition) == 12u);
static_assert(sizeof(PackedTriangle) == 12u);
static_assert(sizeof(MotionTriangleAABB) == 24u);
static_assert(sizeof(MotionMeshValidationStatus) == 16u);

enum MotionMeshValidationError : unsigned int {
    MOTION_MESH_VALIDATION_ERROR_NON_FINITE_POSITION = 1u,
    MOTION_MESH_VALIDATION_ERROR_INVALID_TRIANGLE_INDEX = 2u,
};

[[nodiscard]] __device__ __forceinline__ unsigned int
load_u32(const unsigned char *address) {
    unsigned int value;
    __builtin_memcpy(&value, address, sizeof(value));
    return value;
}

[[nodiscard]] __device__ __forceinline__ float
float_from_bits(unsigned int bits) {
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] __device__ __forceinline__ bool
is_finite_float(unsigned int bits) {
    return (bits & 0x7f800000u) != 0x7f800000u;
}

[[nodiscard]] __device__ __forceinline__ unsigned long long
validation_failure_key(unsigned int error, unsigned long long element,
                       unsigned int component) {
    constexpr auto element_mask = (1ull << 48u) - 1ull;
    return static_cast<unsigned long long>(error) << 56u |
           (element & element_mask) << 8u |
           static_cast<unsigned long long>(component & 0xffu);
}

__device__ __forceinline__ void
report_validation_failure(MotionMeshValidationStatus *status,
                          unsigned long long failure_key) {
    auto address = &status->failure_key;
    auto observed = atomicCAS(address, ~0ull, ~0ull);
    while (failure_key < observed) {
        auto previous = atomicCAS(address, observed, failure_key);
        if (previous == observed) { return; }
        observed = previous;
    }
}

[[nodiscard]] __device__ __forceinline__ float
minimum(float lhs, float rhs) {
    return lhs < rhs ? lhs : rhs;
}

[[nodiscard]] __device__ __forceinline__ float
maximum(float lhs, float rhs) {
    return lhs > rhs ? lhs : rhs;
}

}// namespace

extern "C" __global__ void
pack_motion_vertices(const unsigned char *__restrict__ source_vertices,
                     PackedPosition *__restrict__ packed_vertices,
                     unsigned long long vertex_count,
                     unsigned int source_vertex_stride,
                     MotionMeshValidationStatus *__restrict__ validation_status) {
    auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto grid_stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (; index < vertex_count; index += grid_stride) {
        auto source = source_vertices + index * source_vertex_stride;
        auto x_bits = load_u32(source);
        auto y_bits = load_u32(source + 4u);
        auto z_bits = load_u32(source + 8u);
        auto x_finite = is_finite_float(x_bits);
        auto y_finite = is_finite_float(y_bits);
        auto z_finite = is_finite_float(z_bits);
        if (!x_finite) {
            report_validation_failure(
                validation_status,
                validation_failure_key(
                    MOTION_MESH_VALIDATION_ERROR_NON_FINITE_POSITION,
                    index, 0u));
        }
        if (!y_finite) {
            report_validation_failure(
                validation_status,
                validation_failure_key(
                    MOTION_MESH_VALIDATION_ERROR_NON_FINITE_POSITION,
                    index, 1u));
        }
        if (!z_finite) {
            report_validation_failure(
                validation_status,
                validation_failure_key(
                    MOTION_MESH_VALIDATION_ERROR_NON_FINITE_POSITION,
                    index, 2u));
        }
        packed_vertices[index] = PackedPosition{
            x_finite ? float_from_bits(x_bits) : 0.0f,
            y_finite ? float_from_bits(y_bits) : 0.0f,
            z_finite ? float_from_bits(z_bits) : 0.0f};
    }
}

extern "C" __global__ void
build_motion_triangle_aabbs(
    const unsigned char *__restrict__ source_triangles,
    const PackedPosition *__restrict__ packed_vertices,
    PackedTriangle *__restrict__ packed_triangles,
    MotionTriangleAABB *__restrict__ aabbs,
    unsigned int triangle_count,
    unsigned int vertices_per_keyframe,
    unsigned int keyframe_count,
    MotionMeshValidationStatus *__restrict__ validation_status) {
    auto triangle_index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto grid_stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (; triangle_index < triangle_count; triangle_index += grid_stride) {
        auto source = source_triangles +
                      static_cast<unsigned long long>(triangle_index) *
                          sizeof(PackedTriangle);
        PackedTriangle triangle{
            load_u32(source),
            load_u32(source + 4u),
            load_u32(source + 8u)};
        unsigned int indices[]{triangle.i0, triangle.i1, triangle.i2};
        auto indices_are_valid = true;
        for (auto corner = 0u; corner < 3u; corner++) {
            if (indices[corner] >= vertices_per_keyframe) {
                indices_are_valid = false;
                report_validation_failure(
                    validation_status,
                    validation_failure_key(
                        MOTION_MESH_VALIDATION_ERROR_INVALID_TRIANGLE_INDEX,
                        triangle_index, corner));
            }
        }
        if (!indices_are_valid) {
            // Validation is reported asynchronously after all work already
            // enqueued on the stream completes. Keep the intervening HIPRT
            // build and custom intersection safe by snapshotting a degenerate
            // in-range triangle as well as an empty bound.
            packed_triangles[triangle_index] = PackedTriangle{};
            aabbs[triangle_index] = MotionTriangleAABB{};
            continue;
        }
        packed_triangles[triangle_index] = triangle;

        auto positive_infinity = float_from_bits(0x7f800000u);
        auto negative_infinity = float_from_bits(0xff800000u);
        MotionTriangleAABB bounds{
            positive_infinity,
            positive_infinity,
            positive_infinity,
            negative_infinity,
            negative_infinity,
            negative_infinity};
        for (auto keyframe = 0u; keyframe < keyframe_count; keyframe++) {
            auto keyframe_base =
                static_cast<unsigned long long>(keyframe) *
                vertices_per_keyframe;
            for (auto corner = 0u; corner < 3u; corner++) {
                auto position = packed_vertices[keyframe_base + indices[corner]];
                bounds.min_x = minimum(bounds.min_x, position.x);
                bounds.min_y = minimum(bounds.min_y, position.y);
                bounds.min_z = minimum(bounds.min_z, position.z);
                bounds.max_x = maximum(bounds.max_x, position.x);
                bounds.max_y = maximum(bounds.max_y, position.y);
                bounds.max_z = maximum(bounds.max_z, position.z);
            }
        }
        aabbs[triangle_index] = bounds;
    }
}
