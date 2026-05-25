#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    auto context = luisa::compute::Context{luisa::current_executable_path()};
    auto device = context.create_device("cuda");

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint>(1u);

    using namespace luisa::compute;
    auto shader = device.compile<1>([&](UInt n) noexcept {
        auto z = def(make_uint2(1u));
        $for (i, n) {
            UInt x;
            x = 2u;
            $if (i == 0u) {
                x = 2u;
                $continue;
            };
            n -= 1u;
            $if (n == 1u) {
                x = dispatch_id().x;
                $break;
            };
            z.x *= x + i;
        };
        buffer->write(0u, z.x);
    });

    auto result = 0u;
    stream << shader(10u).dispatch(1u)
           << buffer.copy_to(&result)
           << synchronize();

    LUISA_INFO("result = {}", result);

    auto trace = [&](AccelVar &accel, const UInt &mask, const UInt &ignore) noexcept {
        Var<CommittedHit> hit;
        $if (mask >= 1u) {
            Var<SurfaceHit> h;
            h.inst = ~0u;
            auto inside = mask > 100u;
            $if (!inside) {
                h = accel.intersect(make_ray(make_float3(), make_float3(1.f)), {});
                hit.inst = h.inst;
                hit.prim = h.prim;
                hit.committed_ray_t = h.committed_ray_t;
                hit.bary = h.bary;
                hit.hit_type = ite(h->miss(), static_cast<uint32_t>(HitType::Miss), static_cast<uint32_t>(HitType::Surface));
                $if (h->miss()) {
                    inside = true;
                };
            };
            $if (hit->miss() & inside) {
                $for (i, 100u) {
                    h = accel.intersect(make_ray(make_float3(), make_float3(1.f)), {});
                    $if (hit->miss()) {
                        $break;
                    };
                };
            };
            $if (h->miss()) {
                h = accel.intersect(make_ray(make_float3(), make_float3(1.f)), {});
            };
            hit.inst = h.inst;
            hit.prim = h.prim;
            hit.committed_ray_t = h.committed_ray_t;
            hit.bary = h.bary;
            hit.hit_type = ite(h->miss(), static_cast<uint32_t>(HitType::Miss), static_cast<uint32_t>(HitType::Surface));
        } $else {
        };
        return hit;
    };

    auto shader2 = device.compile<1>([&](AccelVar accel, BufferVar<CommittedHit> hits, UInt mask, BufferUInt buffer) noexcept {
        Var<CommittedHit> hit;
        hit->inst = ~0u;
        $while (true) {
            $if (mask >= 1u) {
                auto h = accel.intersect(make_ray(make_float3(), make_float3(1.f)), {});
                hit.inst = h.inst;
                hit.prim = h.prim;
                hit.committed_ray_t = h.committed_ray_t;
                hit.bary = h.bary;
                hit.hit_type = ite(h->miss(), static_cast<uint32_t>(HitType::Miss), static_cast<uint32_t>(HitType::Surface));
                $if (hit->miss()) {
                    $break;
                };
            } $else {
                hit = trace(accel, mask, hit.prim);
                $if (mask == 2u) {
                    $if (hits->read(0u)->miss()) {
                        $break;
                    };
                    $continue;
                };
            };
            $if (!hit->miss()) {
                hits->write(0u, hit);
            };
        };
        auto v = def(0u);
        $if (true) {
            v = 1u;
        };
        buffer->write(0u, v);
        // $if (mask != 0u) {
        //     $if (!hit->miss()) {
        //         hits->write(0u, hit);
        //     };
        // };
    });
}
