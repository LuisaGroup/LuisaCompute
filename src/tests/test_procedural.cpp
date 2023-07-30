#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <stb/stb_image_write.h>
#include <iostream>

using namespace luisa;
using namespace luisa::compute;

float lcg(uint &state) noexcept {
    constexpr auto lcg_a = 1664525u;
    constexpr auto lcg_c = 1013904223u;
    state = lcg_a * state + lcg_c;
    return cast<float>(state & 0x00ffffffu) *
           (1.0f / static_cast<float>(0x01000000u));
};

int main(int argc, char *argv[]) {
    constexpr uint32_t width = 1280;
    constexpr uint32_t height = 720;
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();
    Image<float> device_image1 = device.create_image<float>(PixelStorage::FLOAT4, width, height);

    int count = 1024;
    float radius = .2f;
    // aabb
    luisa::vector<AABB> aabbs{size_t(count)};
    uint state = 0;
    for (int i = 0; i < count; i++) {
        float3 pos = make_float3(lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f, lcg(state) * 2.f - 1.f) * 10.f;
        float3 max = pos + radius;
        float3 min = pos - radius;
        aabbs[i].packed_max = {max.x, max.y, max.z};
        aabbs[i].packed_min = {min.x, min.y, min.z};
    }
    Accel accel = device.create_accel();
    Buffer<AABB> aabb_buffer = device.create_buffer<AABB>(count);
    ProceduralPrimitive procedural_primitives = device.create_procedural_primitive(aabb_buffer.view());
    accel.emplace_back(procedural_primitives);
    // triangle
    std::array vertices{
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)};
    std::array indices{0u, 1u, 2u};
    Buffer<float3> vertex_buffer = device.create_buffer<float3>(3u);
    Buffer<Triangle> triangle_buffer = device.create_buffer<Triangle>(1u);
    Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    accel.emplace_back(mesh, scaling(5.0f), 0xffu, false);
    stream << aabb_buffer.copy_from(aabbs.data())
           << procedural_primitives.build()
           << vertex_buffer.copy_from(vertices.data())
           << triangle_buffer.copy_from(indices.data())
           << mesh.build()
           << accel.build()
           << synchronize();

    static constexpr uint spp = 1024u;

    Callable tea = [](UInt v0, UInt v1) noexcept {
        UInt s0 = def(0u);
        for (uint n = 0u; n < 4u; n++) {
            s0 += 0x9e3779b9u;
            v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
            v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
        }
        return v0;
    };

    Kernel2D kernel = [&](Float3 pos, UInt frame_id) {
        UInt2 coord = dispatch_id().xy();
        UInt2 size = dispatch_size().xy();
        Float aspect = size.x.cast<float>() / size.y.cast<float>();
        // very bad jitter
        Float2 jitter = make_float2(make_uint2(tea(coord.x, frame_id),
                                             tea(coord.y, frame_id))) /
                      static_cast<float>(~0u);
        Float2 p = (make_float2(coord) + jitter) / make_float2(size) * 2.f - 1.f;
        static constexpr float fov = radians(45.8f);
        Float3 origin = pos;
        Float3 direction = normalize(make_float3(p * tan(0.5f * fov) * make_float2(aspect, 1.0f), -1.0f));
        Var<Ray> ray = make_ray(origin, direction);

        // traversal aceeleration structure with ray-query
        Float3 sphere_color;
        Var<CommittedHit> hit = accel->query_all(ray)
                       .on_triangle_candidate([&](TriangleCandidate &candidate) noexcept {
                           Var<TriangleHit> h = candidate.hit();
                           Float3 uvw = make_float3(1.f - h.bary.x - h.bary.y, h.bary);
                           $if(length(uvw.xy()) < .8f &
                               length(uvw.yz()) < .8f &
                               length(uvw.zx()) < .8f) {
                               candidate.commit();
                           };
                       })
                       .on_procedural_candidate([&](ProceduralCandidate &candidate) noexcept {
                           Var<ProceduralHit> h = candidate.hit();
                           Var<Ray> ray = candidate.ray();
                           Var<AABB> aabb = aabb_buffer->read(h.prim);
                           //ray-sphere intersection
                           Float3 origin = (aabb->min() + aabb->max()) * .5f;
                           Float3 ray_origin = ray->origin();
                           Float3 L = origin - ray_origin;
                           Float3 dir = ray->direction();
                           Float cos_theta = dot(dir, normalize(L));
                           $if(cos_theta > 0.f) {
                               Float d_oc = length(L);
                               Float tc = d_oc * cos_theta;
                               Float d = sqrt(d_oc * d_oc - tc * tc);
                               $if(d <= radius) {
                                   Float t1c = sqrt(radius * radius - d * d);
                                   Float dist = tc - t1c;
                                   // save normal as color
                                   $if(dist <= ray->t_max()) {
                                       Float3 normal = normalize(ray_origin + dir * dist - origin);
                                       sphere_color = normal * 0.5f + 0.5f;
                                   };
                                   candidate.commit(dist);
                               };
                           };
                       })
                       .trace();

        Float3 old = device_image1->read(coord).xyz();
        Float3 color = def(make_float3());
        $if(hit->is_procedural()) {
            color = sphere_color;
        }
        $elif(hit->is_triangle()) {
            color = make_float3(1.f - hit.bary.x - hit.bary.y, hit.bary);
        };
        Float n = cast<float>(frame_id + 1u);
        device_image1->write(coord, make_float4(lerp(old, color, 1.f / n), 1.f));
    };

    Image<float> ldr_image = device.create_image<float>(PixelStorage::BYTE4, width, height);

    auto clear = device.compile<2>([](ImageFloat image) noexcept {
        UInt2 coord = dispatch_id().xy();
        image.write(coord, make_float4());
    });

    auto blit = device.compile<2>([](ImageFloat image_in, ImageFloat image_out) noexcept {
        UInt2 p = dispatch_id().xy();
        image_out.write(p, image_in.read(p));
    });

    luisa::vector<std::array<uint8_t, 4u>> pixels{width * height};
    auto s = device.compile(kernel);
    const float3 pos = make_float3(0.f, 0.f, 18.0f);
    CommandList list;
    list.reserve(spp, 0u);
    for (uint i = 0u; i < spp; i++) {
        list << s(pos, i).dispatch(width, height);
    }
    Clock clk;
    stream << clear(device_image1).dispatch(width, height)
           << [&clk] { clk.tic(); }
           << list.commit()
           << [&clk] { LUISA_INFO("Rendering finished in {} ms.", clk.toc()); }
           << blit(device_image1, ldr_image).dispatch(width, height)
           << ldr_image.copy_to(pixels.data())
           << synchronize();
    stbi_write_png("test_procedural.png", width, height, 4, pixels.data(), 0);
}

