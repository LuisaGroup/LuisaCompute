// Hardware ray tracing vs the software (fallback) BVH, on one scene.
//
// The fallback acceleration structure is a property of the *device*, chosen at
// creation time from the backend's `DeviceConfigExt::use_fallback_rtx()` (or
// auto-enabled on a platform without hardware ray tracing).  So the comparison
// this test has to make is between two devices:
//
//   configuration A - fallback disabled: the native path (OptiX / DXR / VKRT)
//   configuration B - fallback enabled : the software LBVH of
//                     src/backends/common/rtx (luisa-fallback-rtx)
//
// Both devices are handed the *same* scene and the *same* rays, and the results
// must agree.  The scene and the rays are generated on the host from a fixed
// seed, so the only variable between the two runs is the acceleration-structure
// implementation itself.
//
// What "agree" means, and why it is not simply `==`:
//
// * hit vs miss, the hit instance, the hit primitive, the instance user id and
//   the instance visibility mask are compared **exactly**.  Those are the parts
//   of the result an application acts on, and a mismatch there is a bug.
// * the hit distance is compared with a small relative tolerance, and the
//   maximum observed error is printed.  The hardware ray/triangle test is not
//   bit-identical to a float `Möller-Trumbore` evaluation (it is a
//   mixed fixed/float-precision slab + Moller-Trumbore pipeline), so a
//   last-few-ULP distance difference on grazing rays is inherent to the
//   *hardware* path, not to the fallback: the fallback's own distance is
//   bit-identical to the independent software reference that
//   `examples/compute/software_lbvh.cpp` cross-checks (it reports
//   `max relative distance error 0.000e+00` when the fallback answers the RTX
//   query).  The tolerance here is what that demo uses for the hardware path.
// * barycentrics are compared with a small absolute tolerance (same reason).
//
// The test runs the two configurations sequentially in one process and destroys
// the first device before creating the second, which is the documented
// constraint for the Vulkan backend (Vulkan's loader dispatch is process-global,
// so at most one Vulkan Device may be alive at a time).
//
// Usage: test_fallback_rtx <backend> [--quick]

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#if defined(LUISA_TEST_FALLBACK_RTX_HAS_CUDA)
#include <luisa/backends/ext/cuda/cuda_config_ext.h>
#endif
#if defined(LUISA_TEST_FALLBACK_RTX_HAS_DX)
#include <luisa/backends/ext/dx_config_ext.h>
#endif
#if defined(LUISA_TEST_FALLBACK_RTX_HAS_VK)
#include <luisa/backends/ext/vk_config_ext.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto invalid_index = std::numeric_limits<uint32_t>::max();

// The per-ray record the trace kernel writes.  The fields are kept in separate
// buffers rather than in one struct so that the kernel stays free of struct
// locals, which is the part of the DSL most likely to differ between backends.
struct TraceOutput {
    luisa::vector<uint> flags;    // bit 0: closest hit, bit 1: any-hit hit
    luisa::vector<uint> instance; // invalid_index on a miss
    luisa::vector<uint> primitive;
    luisa::vector<float> distance; // -1 on a miss
    luisa::vector<float2> bary;
    luisa::vector<uint> user_id;
    luisa::vector<uint> visibility;
};

// ---------------------------------------------------------------------------
// Scene: host-side geometry, instances and rays (deterministic, fixed seed).
// ---------------------------------------------------------------------------
struct Scene {
    luisa::vector<float3> vertices;
    luisa::vector<Triangle> triangles;
    struct Mesh {
        uint triangle_offset;
        uint triangle_count;
    };
    luisa::vector<Mesh> meshes;
    struct Instance {
        uint mesh;
        float4x4 to_world;
        uint visibility;
        uint user_id;
    };
    luisa::vector<Instance> instances;
    luisa::vector<Ray> rays;
    luisa::vector<uint> ray_masks;
};

void add_mesh(Scene &scene, luisa::span<const float3> vertices,
              luisa::span<const uint3> triangles) noexcept {
    auto vertex_base = static_cast<uint>(scene.vertices.size());
    auto triangle_base = static_cast<uint>(scene.triangles.size());
    for (auto v : vertices) { scene.vertices.emplace_back(v); }
    for (auto t : triangles) {
        scene.triangles.emplace_back(Triangle{t.x + vertex_base,
                                              t.y + vertex_base,
                                              t.z + vertex_base});
    }
    scene.meshes.emplace_back(Scene::Mesh{triangle_base,
                                          static_cast<uint>(triangles.size())});
}

// A unit cube in [-0.5, 0.5]^3: the well-conditioned bulk of the scene.
void add_cube(Scene &scene) noexcept {
    static const float3 vertices[] = {
        float3(-0.5f, -0.5f, -0.5f), float3(0.5f, -0.5f, -0.5f),
        float3(-0.5f, 0.5f, -0.5f), float3(0.5f, 0.5f, -0.5f),
        float3(-0.5f, -0.5f, 0.5f), float3(0.5f, -0.5f, 0.5f),
        float3(-0.5f, 0.5f, 0.5f), float3(0.5f, 0.5f, 0.5f)};
    static const uint3 triangles[] = {
        uint3(0u, 2u, 1u), uint3(1u, 2u, 3u), uint3(4u, 5u, 6u), uint3(6u, 5u, 7u),
        uint3(0u, 4u, 2u), uint3(2u, 4u, 6u), uint3(1u, 3u, 5u), uint3(5u, 3u, 7u),
        uint3(0u, 1u, 4u), uint3(4u, 1u, 5u), uint3(2u, 6u, 3u), uint3(3u, 6u, 7u)};
    add_mesh(scene, luisa::span{vertices}, luisa::span{triangles});
}

// A quad in the z = 0 plane: overlapping instances, so many rays hit two meshes
// and the choice of the closest one actually matters.
void add_quad(Scene &scene) noexcept {
    static const float3 vertices[] = {
        float3(-1.0f, -1.0f, 0.0f), float3(1.0f, -1.0f, 0.0f),
        float3(-1.0f, 1.0f, 0.0f), float3(1.0f, 1.0f, 0.0f)};
    static const uint3 triangles[] = {uint3(0u, 1u, 2u), uint3(2u, 1u, 3u)};
    add_mesh(scene, luisa::span{vertices}, luisa::span{triangles});
}

// A degenerate, zero-area sliver: the LBVH has to survive it and the two paths
// have to agree that it is (not) hit the same way.
void add_sliver(Scene &scene) noexcept {
    static const float3 vertices[] = {
        float3(-0.5f, 0.0f, 0.0f), float3(0.5f, 1.0e-4f, 0.0f), float3(0.0f, 0.0f, 0.0f)};
    static const uint3 triangles[] = {uint3(0u, 1u, 2u)};
    add_mesh(scene, luisa::span{vertices}, luisa::span{triangles});
}

[[nodiscard]] Scene make_scene(uint ray_count) noexcept {
    Scene scene;
    add_cube(scene);
    add_quad(scene);
    add_sliver(scene);
    // Instances: cubes inside a lattice, quad instances crossing it (so meshes
    // overlap and the closest-hit ordering is exercised), one rotated quad
    // behind the lattice, plus two instances that the visibility mask must be
    // able to exclude and one carrying a user id.
    for (auto i = 0u; i < 3u; i++) {
        for (auto j = 0u; j < 3u; j++) {
            auto x = (static_cast<float>(i) - 1.0f) * 1.4f;
            auto y = (static_cast<float>(j) - 1.0f) * 1.4f;
            scene.instances.emplace_back(Scene::Instance{
                0u, translation(make_float3(x, y, 0.0f)) * scaling(0.9f),
                0xffu, 0u});
        }
    }
    scene.instances.emplace_back(Scene::Instance{
        1u, translation(make_float3(0.0f, 0.0f, -2.0f)) * scaling(1.5f), 0xffu, 7u});
    scene.instances.emplace_back(Scene::Instance{
        1u, translation(make_float3(0.0f, 0.0f, 2.0f)) *
                rotation(make_float3(1.0f, 0.0f, 0.0f), radians(90.0f)),
        0xffu, 0u});
    // Only visible to the rays whose mask carries bit 0.
    scene.instances.emplace_back(Scene::Instance{
        0u, translation(make_float3(0.0f, 0.0f, 3.0f)) * scaling(0.5f), 0x01u, 11u});
    // Never visible: no ray mask can select it.
    scene.instances.emplace_back(Scene::Instance{
        0u, translation(make_float3(0.0f, 0.0f, -3.0f)) * scaling(0.5f), 0x00u, 13u});
    scene.instances.emplace_back(Scene::Instance{
        2u, translation(make_float3(0.9f, 0.9f, 0.35f)) *
                rotation(make_float3(0.0f, 1.0f, 0.0f), radians(30.0f)),
        0xffu, 0u});

    // Rays: a jittered grid looking down -z through the scene, and a pseudo
    // random cloud around it.  A few of them start inside the scene (t_min > 0)
    // and a few carry a partial visibility mask.
    uint32_t state = 0x1234'5678u;
    auto random = [&state]() noexcept {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8u) * (1.0f / 16777216.0f);
    };
    auto grid = ray_count / 2u;
    auto side = static_cast<uint>(std::sqrt(static_cast<double>(grid)));
    if (side == 0u) { side = 1u; }
    for (auto y = 0u; y < side; y++) {
        for (auto x = 0u; x < side; x++) {
            auto u = (static_cast<float>(x) + random() - 0.5f) / static_cast<float>(side);
            auto v = (static_cast<float>(y) + random() - 0.5f) / static_cast<float>(side);
            auto origin = make_float3((u - 0.5f) * 6.0f, (v - 0.5f) * 6.0f, 5.0f);
            auto direction = normalize(make_float3((u - 0.5f) * 0.6f,
                                                   (v - 0.5f) * 0.6f, -1.0f));
            scene.rays.emplace_back(Ray{{origin.x, origin.y, origin.z}, 0.0f,
                                        {direction.x, direction.y, direction.z}, 100.0f});
            scene.ray_masks.emplace_back(0xffu);
        }
    }
    while (scene.rays.size() < ray_count) {
        auto index = static_cast<uint>(scene.rays.size());
        auto origin = make_float3((random() - 0.5f) * 8.0f,
                                  (random() - 0.5f) * 8.0f,
                                  (random() - 0.5f) * 8.0f);
        auto target = make_float3((random() - 0.5f) * 3.0f,
                                  (random() - 0.5f) * 3.0f,
                                  (random() - 0.5f) * 3.0f);
        auto direction = normalize(target - origin);
        // every 8th ray starts inside the scene, every 4th carries a partial mask
        auto t_min = index % 8u == 0u ? 0.5f : 0.0f;
        scene.rays.emplace_back(Ray{{origin.x, origin.y, origin.z}, t_min,
                                    {direction.x, direction.y, direction.z}, 100.0f});
        scene.ray_masks.emplace_back(index % 4u == 0u ? 0x01u : 0xffu);
    }
    scene.rays.resize(ray_count);
    scene.ray_masks.resize(ray_count);
    return scene;
}

// ---------------------------------------------------------------------------
// Device configuration: the fallback is switched through the backend's
// DeviceConfigExt, which is the supported surface.
// ---------------------------------------------------------------------------
[[nodiscard]] luisa::unique_ptr<DeviceConfigExt>
make_config_ext(luisa::string_view backend, bool fallback) noexcept {
#if defined(LUISA_TEST_FALLBACK_RTX_HAS_CUDA)
    if (backend == "cuda") {
        struct Ext final : CUDADeviceConfigExt {
            bool fallback;
            explicit Ext(bool f) noexcept : fallback{f} {}
            [[nodiscard]] bool use_fallback_rtx() const noexcept override { return fallback; }
        };
        return luisa::make_unique<Ext>(fallback);
    }
#endif
#if defined(LUISA_TEST_FALLBACK_RTX_HAS_DX)
    if (backend == "dx") {
        struct Ext final : DirectXDeviceConfigExt {
            bool fallback;
            explicit Ext(bool f) noexcept : fallback{f} {}
            [[nodiscard]] bool use_fallback_rtx() const noexcept override { return fallback; }
        };
        return luisa::make_unique<Ext>(fallback);
    }
#endif
#if defined(LUISA_TEST_FALLBACK_RTX_HAS_VK)
    if (backend == "vk") {
        struct Ext final : VulkanDeviceConfigExt {
            bool fallback;
            explicit Ext(bool f) noexcept : fallback{f} {}
            [[nodiscard]] bool use_fallback_rtx() const noexcept override { return fallback; }
        };
        return luisa::make_unique<Ext>(fallback);
    }
#endif
    static_cast<void>(backend);
    static_cast<void>(fallback);
    return nullptr;
}

// The trace kernel: closest hit and any-hit for the same ray, plus the instance
// properties of the hit.  Everything goes through buffers so the host side can
// compare the two configurations field by field.
[[nodiscard]] auto make_trace_kernel() noexcept {
    return Kernel1D{[](AccelVar accel, BufferVar<Ray> rays, BufferVar<uint> masks,
              BufferVar<uint> flags, BufferVar<uint> instance, BufferVar<uint> primitive,
              BufferVar<float> distance, BufferVar<float2> bary,
              BufferVar<uint> user_id, BufferVar<uint> visibility,
              UInt ray_count) noexcept {
        set_block_size(64u);
        UInt i = dispatch_id().x;
        $if (i < ray_count) {
            auto ray = rays.read(i);
            auto mask = masks.read(i);
            auto closest = accel.intersect(ray, AccelTraceOptions{.visibility_mask = mask});
            auto any = accel.intersect_any(ray, AccelTraceOptions{.visibility_mask = mask});
            auto hit = def(false);
            auto out_instance = def(invalid_index);
            auto out_primitive = def(invalid_index);
            auto out_distance = def(-1.0f);
            auto out_bary = def(make_float2(0.0f));
            auto out_user_id = def(0u);
            auto out_visibility = def(0u);
            $if (!closest->miss()) {
                hit = true;
                out_instance = closest.inst;
                out_primitive = closest.prim;
                out_distance = closest.committed_ray_t;
                out_bary = closest.bary;
                // only meaningful for a hit; the property reads must not be
                // evaluated for a miss (the instance index would be ~0)
                out_user_id = accel.instance_user_id(closest.inst);
                out_visibility = accel.instance_visibility_mask(closest.inst);
            };
            flags.write(i, cast<uint>(hit) | (cast<uint>(any) << 1u));
            instance.write(i, out_instance);
            primitive.write(i, out_primitive);
            distance.write(i, out_distance);
            bary.write(i, out_bary);
            user_id.write(i, out_user_id);
            visibility.write(i, out_visibility);
        };
    }};
}

// One full device lifetime: create the device with the requested configuration,
// build the scene, trace the rays, read the results back, and destroy the
// device (the caller must not hold on to anything of it).
[[nodiscard]] TraceOutput run_configuration(Context &context, luisa::string_view backend,
                                            bool fallback, const Scene &scene) noexcept {
    DeviceConfig config;
    config.extension = make_config_ext(backend, fallback);
    auto has_config_hook = config.extension != nullptr;
    Device device = context.create_device(backend, &config);
    Stream stream = device.create_stream();

    // Geometry -------------------------------------------------------------
    Buffer<float3> vertices = device.create_buffer<float3>(scene.vertices.size());
    Buffer<Triangle> triangles = device.create_buffer<Triangle>(scene.triangles.size());
    Buffer<Ray> rays = device.create_buffer<Ray>(scene.rays.size());
    Buffer<uint> masks = device.create_buffer<uint>(scene.ray_masks.size());
    auto ray_count = static_cast<uint>(scene.rays.size());
    Buffer<uint> flags = device.create_buffer<uint>(ray_count);
    Buffer<uint> instance = device.create_buffer<uint>(ray_count);
    Buffer<uint> primitive = device.create_buffer<uint>(ray_count);
    Buffer<float> distance = device.create_buffer<float>(ray_count);
    Buffer<float2> bary = device.create_buffer<float2>(ray_count);
    Buffer<uint> user_id = device.create_buffer<uint>(ray_count);
    Buffer<uint> visibility = device.create_buffer<uint>(ray_count);

    luisa::vector<Mesh> meshes;
    meshes.reserve(scene.meshes.size());
    for (auto &&mesh : scene.meshes) {
        meshes.emplace_back(device.create_mesh(
            vertices, triangles.view(mesh.triangle_offset, mesh.triangle_count)));
    }
    Accel accel = device.create_accel();
    for (auto &&i : scene.instances) {
        accel.emplace_back(meshes[i.mesh], i.to_world,
                           static_cast<uint8_t>(i.visibility), true, i.user_id);
    }

    stream << vertices.copy_from(luisa::span{scene.vertices})
           << triangles.copy_from(luisa::span{scene.triangles})
           << rays.copy_from(luisa::span{scene.rays})
           << masks.copy_from(luisa::span{scene.ray_masks});
    for (auto &&mesh : meshes) { stream << mesh.build(); }
    stream << accel.build() << synchronize();

    // Trace ----------------------------------------------------------------
    auto shader = device.compile(make_trace_kernel());
    stream << shader(accel, rays, masks, flags, instance, primitive,
                     distance, bary, user_id, visibility, ray_count)
                  .dispatch(ray_count);

    TraceOutput output;
    output.flags.resize(ray_count);
    output.instance.resize(ray_count);
    output.primitive.resize(ray_count);
    output.distance.resize(ray_count);
    output.bary.resize(ray_count);
    output.user_id.resize(ray_count);
    output.visibility.resize(ray_count);
    stream << flags.copy_to(luisa::span{output.flags})
           << instance.copy_to(luisa::span{output.instance})
           << primitive.copy_to(luisa::span{output.primitive})
           << distance.copy_to(luisa::span{output.distance})
           << bary.copy_to(luisa::span{output.bary})
           << user_id.copy_to(luisa::span{output.user_id})
           << visibility.copy_to(luisa::span{output.visibility})
           << synchronize();
    LUISA_INFO("  configuration '{}': {} rays traced (config ext {})",
               fallback ? "fallback" : "hardware", ray_count,
               has_config_hook ? "supplied" : "absent");
    return output;
}

struct Comparison {
    size_t flag_mismatch{};
    size_t instance_mismatch{};
    size_t primitive_mismatch{};
    size_t user_id_mismatch{};
    size_t visibility_mismatch{};
    size_t bary_mismatch{};
    size_t distance_mismatch{};
    size_t hits{};
    double max_relative_distance_error{};
    double max_barycentric_error{};
};

[[nodiscard]] Comparison compare(const TraceOutput &a, const TraceOutput &b) noexcept {
    Comparison c;
    for (auto i = 0u; i < a.flags.size(); i++) {
        c.flag_mismatch += a.flags[i] != b.flags[i];
        c.instance_mismatch += a.instance[i] != b.instance[i];
        c.primitive_mismatch += a.primitive[i] != b.primitive[i];
        c.user_id_mismatch += a.user_id[i] != b.user_id[i];
        c.visibility_mismatch += a.visibility[i] != b.visibility[i];
        c.hits += (a.flags[i] & 1u) != 0u;
        if (a.instance[i] != invalid_index && b.instance[i] != invalid_index) {
            auto x = static_cast<double>(a.distance[i]);
            auto y = static_cast<double>(b.distance[i]);
            auto scale = std::max({std::abs(x), std::abs(y), 1.0e-6});
            c.max_relative_distance_error =
                std::max(c.max_relative_distance_error, std::abs(x - y) / scale);
            auto du = std::abs(static_cast<double>(a.bary[i].x) - static_cast<double>(b.bary[i].x));
            auto dv = std::abs(static_cast<double>(a.bary[i].y) - static_cast<double>(b.bary[i].y));
            c.max_barycentric_error = std::max({c.max_barycentric_error, du, dv});
            c.bary_mismatch += du > 5.0e-3 || dv > 5.0e-3;
            c.distance_mismatch += std::abs(x - y) / scale > 1.0e-3;
        }
    }
    return c;
}

}// namespace

int main(int argc, char *argv[]) {
    auto exe = luisa::test::safe_argv0();
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        luisa::test::print_device_usage(exe);
        return 1;
    }
    luisa::string backend = argv[1];
    auto quick = false;
    for (auto i = 2; i < argc; i++) {
        if (luisa::string_view{argv[i]} == "--quick") { quick = true; }
    }
    auto ray_count = quick ? 2048u : 8192u;
    auto scene = make_scene(ray_count);
    LUISA_INFO("fallback RTX comparison on '{}': {} vertices, {} triangles, "
               "{} meshes, {} instances, {} rays",
               backend, scene.vertices.size(), scene.triangles.size(),
               scene.meshes.size(), scene.instances.size(), scene.rays.size());

    // A backend whose config extension is not compiled in cannot force the
    // fallback, so there is nothing to compare: report and skip.
    if (make_config_ext(backend, true) == nullptr) {
        LUISA_WARNING("backend '{}' has no fallback-RTX DeviceConfigExt in this build; "
                      "skipping the comparison",
                      backend);
        return 0;
    }

    Context context{exe};
    // Sequential device lifetimes: the Vulkan loader dispatch table is
    // process-global, so the two configurations must not be alive together.
    auto hardware = run_configuration(context, backend, false, scene);
    auto fallback = run_configuration(context, backend, true, scene);

    auto c = compare(hardware, fallback);
    LUISA_INFO("hardware : {} hits of {} rays", c.hits, hardware.flags.size());
    LUISA_INFO("comparison: hit/miss {}, instance {}, primitive {}, user id {}, visibility {}",
               c.flag_mismatch, c.instance_mismatch, c.primitive_mismatch,
               c.user_id_mismatch, c.visibility_mismatch);
    LUISA_INFO("comparison: distance {} (max relative error {:.3e}), "
               "barycentric {} (max error {:.3e})",
               c.distance_mismatch, c.max_relative_distance_error,
               c.bary_mismatch, c.max_barycentric_error);

    expect(c.hits > 0u) << "the scene must produce hits, otherwise the test proves nothing";
    expect(c.flag_mismatch == 0u) << "hit/miss and any-hit disagree between hardware and fallback";
    expect(c.instance_mismatch == 0u) << "the hit instance disagrees";
    expect(c.primitive_mismatch == 0u) << "the hit primitive disagrees";
    expect(c.user_id_mismatch == 0u) << "the instance user id disagrees";
    expect(c.visibility_mismatch == 0u) << "the instance visibility mask disagrees";
    expect(c.distance_mismatch == 0u) << "the hit distance disagrees beyond the tolerance";
    expect(c.bary_mismatch == 0u) << "the hit barycentrics disagree beyond the tolerance";
    return 0;
}
