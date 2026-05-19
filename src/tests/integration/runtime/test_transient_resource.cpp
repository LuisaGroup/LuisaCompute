#include "ut/ut.hpp"
#include "test_device.h"

#include "transient_resource_device/transient_resource_device.h"
#include "reference_image.h"
#include <filesystem>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/gui/window.h>
using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_transient_resource(Device &device) {
    log_level_verbose();

    auto argv = boost::ut::detail::cfg::largv;
    Context context{argv[0]};
    auto opts = luisa::test::ImageTestOptions::parse(
        boost::ut::detail::cfg::largc,
        boost::ut::detail::cfg::largv);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    auto write_shader = device.compile<2>([](ImageVar<float> img, UInt2 offset, Float z_value) {
        auto uv = (make_float2(dispatch_id().xy()) + 0.5f) / make_float2(dispatch_size().xy());
        img.write(dispatch_id().xy() + offset, make_float4(uv, z_value, 1.0f));
    });
    auto write_buffer = device.compile<1>([](BufferVar<float> buffer, BufferVar<float> buffer1) {
        buffer.write(dispatch_id().x, dispatch_id().x.cast<float>());
        buffer1.write(dispatch_id().x, dispatch_id().x.cast<float>());
    });
    static constexpr uint2 resolution = make_uint2(1024u);
    if (!opts.offline) {
        Window window{"path tracing", resolution};
        Swapchain swap_chain = device.create_swapchain(
            stream,
            SwapchainOption{
                .display = window.native_display(),
                .window = window.native_handle(),
                .size = make_uint2(resolution),
                .wants_hdr = false,
                .wants_vsync = false,
                .back_buffer_count = 8,
            });
        luisa::compute::Device transient_res_device{luisa::make_unique<utils::TransientResourceDevice>(Context{context}, device.impl())};
        auto storage = swap_chain.backend_storage();
        auto dst_tex = device.create_image<float>(storage, resolution);
        {
            utils::TransientResourceDeviceScope managed_scope{
                stream,
                transient_res_device,
                true};
            // pass 0
            {
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                managed_scope.cmdlist << write_shader(tex, uint2(0), 0.0f).dispatch(resolution.x, resolution.y / 2);
            }
            // pass 1
            {
                // exactly same texture as pass 0
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                managed_scope.cmdlist << write_shader(tex, uint2(0, resolution.y / 2), 1.0f).dispatch(resolution.x, resolution.y / 2);
            }
            // pass 2
            {
                // exactly same texture as pass 0 and pass 1
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                // mix-in usage of transient and REAL resources
                managed_scope.cmdlist << dst_tex.copy_from(tex);
            }
            // pass 3
            {
                auto buffer = managed_scope.create_transient_buffer<float>("MyBuffer", 512);
                auto buffer1 = managed_scope.create_transient_buffer<float>("MyBuffer1", 256);
                auto buffer2 = managed_scope.create_transient_buffer<float>("MyBuffer2", 384);
                managed_scope.cmdlist
                    << write_buffer(buffer, buffer1).dispatch(buffer.size())
                    // buffer2 will reuse buffer's memory
                    << write_buffer(buffer1, buffer2).dispatch(buffer.size());
            }
            // dispatch to stream after scope
        }
        while (!window.should_close()) {
            window.poll_events();
            stream << swap_chain.present(dst_tex);
        }
        stream.synchronize();
        return;
    } else {
        luisa::compute::Device transient_res_device{luisa::make_unique<utils::TransientResourceDevice>(Context{context}, device.impl())};
        auto storage = PixelStorage::BYTE4;
        auto dst_tex = device.create_image<float>(storage, resolution);
        {
            utils::TransientResourceDeviceScope managed_scope{
                stream,
                transient_res_device,
                true};
            {
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                managed_scope.cmdlist << write_shader(tex, uint2(0), 0.0f).dispatch(resolution.x, resolution.y / 2);
            }
            {
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                managed_scope.cmdlist << write_shader(tex, uint2(0, resolution.y / 2), 1.0f).dispatch(resolution.x, resolution.y / 2);
            }
            {
                auto tex = managed_scope.create_transient_image<float>("MyTexture", storage, resolution);
                managed_scope.cmdlist << dst_tex.copy_from(tex);
            }
            {
                auto buffer = managed_scope.create_transient_buffer<float>("MyBuffer", 512);
                auto buffer1 = managed_scope.create_transient_buffer<float>("MyBuffer1", 256);
                auto buffer2 = managed_scope.create_transient_buffer<float>("MyBuffer2", 384);
                managed_scope.cmdlist
                    << write_buffer(buffer, buffer1).dispatch(buffer.size())
                    << write_buffer(buffer1, buffer2).dispatch(buffer.size());
            }
        }
        luisa::vector<std::byte> pixels(dst_tex.view().size_bytes());
        stream << dst_tex.copy_to(luisa::span{pixels}) << synchronize();
        if (opts.compare_path) {
            auto result = luisa::test::compare_with_reference_file(
                reinterpret_cast<const uint8_t *>(pixels.data()), static_cast<int>(resolution.x), static_cast<int>(resolution.y), 4,
                *opts.compare_path);
            LUISA_INFO("Reference comparison [test_transient_resource]: {} ({})", result.passed ? "PASSED" : "FAILED", result.message);
            if (!result.passed) {
            boost::ut::expect(static_cast<bool>(result.passed)) << result.message;
            return;
        }
        }
        return;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char**>(argv));
    auto &device = dc->device;
    test_transient_resource(device);
}
