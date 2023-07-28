#include <atomic>
#include <numbers>
#include <numeric>
#include <algorithm>

#include <stb/stb_image_write.h>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/sugar.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/gui/window.h>

using namespace luisa;
using namespace luisa::compute;

// Credit: https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py
int main(int argc, char *argv[]) {

    Kernel2D render_kernel = [&](ImageFloat display_image) noexcept {
        set_block_size(16u, 8u, 1u);
        auto uv = make_float2(dispatch_id().xy()) /
                  make_float2(dispatch_size().xy());
        auto c = def(make_float3());
        c.x = uv.x;
        c.y = uv.y;
        c.z = .5f;
        display_image.write(dispatch_id().xy(),
                            make_float4(c, 1.f));
    };

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    auto render_kernel_ir = AST2IR::build_kernel(render_kernel.function()->function());
    auto render = device.compile<2, Image<float>>(render_kernel_ir->get());

    static constexpr auto width = 1280u;
    static constexpr auto height = 720u;
    auto image = device.create_image<float>(PixelStorage::BYTE4, width, height);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"Display", width, height};
    auto swap_chain{device.create_swapchain(
        window.native_handle(), stream,
        make_uint2(width, height),
        false, false, 2)};

    while (!window.should_close()) {
        stream << render(image).dispatch(width, height)
               << swap_chain.present(image);
        window.poll_events();
    }

    stream << synchronize();
}

