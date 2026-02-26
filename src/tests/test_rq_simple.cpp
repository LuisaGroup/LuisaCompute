#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{Triangle{0u, 1u, 2u}};
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    
    Buffer<AABB> aabb_buffer = device.create_buffer<AABB>(1u);
    ProceduralPrimitive procedural = device.create_procedural_primitive(aabb_buffer);

    Accel accel = device.create_accel();
    accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, false);
    accel.emplace_back(procedural, translation(make_float3(0.0f, 0.0f, 5.0f)), 0xffu, false);
    
    std::array aabbs{AABB{.packed_min = {-1.0f, -1.0f, -1.0f}, .packed_max = {1.0f, 1.0f, 1.0f}}};

    stream << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data())
           << mesh.build()
           << aabb_buffer.copy_from(aabbs.data())
           << procedural.build()
           << accel.build()
           << synchronize();

    Buffer<uint> result_buffer = device.create_buffer<uint>(4u);

    Kernel1D kernel = [&]() {
        Var<Ray> ray = make_ray(make_float3(0.0f, 0.0f, -10.0f), make_float3(0.0f, 0.0f, 1.0f));
        auto hit = accel->traverse(ray, {})
            .on_surface_candidate([&](SurfaceCandidate &candidate) noexcept {
                result_buffer->write(0u, 1u);
                // Do not commit triangle
            })
            .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                result_buffer->write(1u, 1u);
                candidate.commit(15.0f);
            })
            .trace();
        result_buffer->write(2u, cast<uint>(hit.hit_type));
        result_buffer->write(3u, hit.inst);
    };

    auto shader = device.compile(kernel);
    std::vector<uint> results(4u, 0u);
    stream << result_buffer.copy_from(results.data())
           << shader().dispatch(1u)
           << result_buffer.copy_to(results.data())
           << synchronize();

    std::cout << "Triangle candidate visited: " << (results[0] ? "YES" : "NO") << std::endl;
    std::cout << "Procedural candidate visited: " << (results[1] ? "YES" : "NO") << std::endl;
    std::cout << "Final hit type: " << results[2] << " (0: MISS, 1: TRIANGLE, 2: PROCEDURAL)" << std::endl;
    std::cout << "Final hit inst: " << results[3] << std::endl;
    return (results[0] == 1u && results[1] == 1u) ? 0 : -1;
}
