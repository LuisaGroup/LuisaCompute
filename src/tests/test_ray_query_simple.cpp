// Minimal test for ray query functionality
// Can be run with: LUISA_EXPERIMENTAL_LLVM_CODEGEN=1 ./test_ray_query_simple cuda

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) {
        std::cerr << "Usage: " << argv[0] << " <backend>\n";
        std::cerr << "  <backend>: cuda, cpu, metal\n";
        return 1;
    }

    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();

    // Create a simple triangle
    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};

    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    Accel accel = device.create_accel();
    accel.emplace_back(mesh);

    stream << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data())
           << mesh.build()
           << accel.build()
           << synchronize();

    // Test 1: Basic triangle ray query
    // This tests the fundamental ray query mechanism
    Buffer<float4> result_buffer = device.create_buffer<float4>(1);

    Kernel1D test_basic_triangle = [&](AccelVar accel, BufferFloat4 result) {
        UInt tid = dispatch_id().x;

        // Simple ray from origin looking down -Z
        Var<Ray> ray = make_ray(make_float3(0.0f, 0.0f, 5.0f), make_float3(0.0f, 0.0f, -1.0f));

        Var<CommittedHit> hit = accel->traverse(ray, {})
                                    .on_surface_candidate([&](SurfaceCandidate &candidate) {
                                        // Commit any triangle hit
                                        candidate.commit();
                                    })
                                    .trace();

        // Store result: (bary.x, bary.y, t, hit_type)
        Float2 bary = hit->bary;
        Float t = hit->committed_ray_t;
        UInt hit_type = hit->hit_type;
        result->write(tid, make_float4(bary.x, bary.y, t, hit_type.cast<float>()));
    };

    auto shader = device.compile(test_basic_triangle);
    stream << shader(accel, result_buffer).dispatch(1)
           << synchronize();

    // Check result
    std::vector<float4> result(1);
    stream << result_buffer.copy_to(result.data())
           << synchronize();

    std::cout << "Test 1 - Basic Triangle Ray Query:\n";
    std::cout << "  Barycentric coords: (" << result[0].x << ", " << result[0].y << ")\n";
    std::cout << "  Hit distance t: " << result[0].z << "\n";
    std::cout << "  Hit kind: " << result[0].w << "\n";

    // Basic validation
    bool success = (result[0].z > 0.0f) && (result[0].z < 10.0f);// Should hit between 0 and 10 units
    if (success) {
        std::cout << "  [PASS] Ray successfully hit triangle\n";
    } else {
        std::cout << "  [FAIL] Ray did not hit triangle as expected\n";
        return 1;
    }

    // Test 2: Ray query with condition
    // This tests control flow within ray query handlers
    Buffer<float> condition_result = device.create_buffer<float>(1);

    Kernel1D test_condition = [&](AccelVar accel, BufferFloat result) {
        UInt tid = dispatch_id().x;

        Var<Ray> ray = make_ray(make_float3(0.0f, 0.0f, 5.0f), make_float3(0.0f, 0.0f, -1.0f));

        Float t_hit = def(0.0f);
        Var<CommittedHit> hit = accel->traverse(ray, {})
                                    .on_surface_candidate([&](SurfaceCandidate &candidate) {
                                        // Only commit if barycentric coords are within certain range
                                        Var<SurfaceHit> h = candidate.hit();
                                        $if ((h.bary.x > 0.1f) & (h.bary.y > 0.1f)) {
                                            candidate.commit();
                                            t_hit = h.committed_ray_t;
                                        };
                                    })
                                    .trace();

        result->write(tid, t_hit);
    };

    auto shader2 = device.compile(test_condition);
    stream << shader2(accel, condition_result).dispatch(1)
           << synchronize();

    std::vector<float> result2(1);
    stream << condition_result.copy_to(result2.data())
           << synchronize();

    std::cout << "\nTest 2 - Ray Query with Condition:\n";
    std::cout << "  Hit distance: " << result2[0] << "\n";

    // If condition filtering worked, we might get a different t or no hit
    // For now just check it ran
    std::cout << "  [INFO] Conditional ray query completed\n";

    std::cout << "\nAll tests completed.\n";
    return 0;
}