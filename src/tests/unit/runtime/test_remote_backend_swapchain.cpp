// Native local-presentation integration test for the C++ remote backend.
//
// A separately launched server executes the image kernel while the client owns
// the Window and platform swapchain. The test presents several finite frames,
// synchronizes both sides of the bridge, and validates the remote image data.

#include "ut/ut.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <system_error>

#include <luisa/backends/ext/remote_config_ext.h>
#include <luisa/gui/window.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_local_swapchain(const char *program_path,
                          string_view local_backend,
                          string_view host, uint16_t port,
                          string_view token) {
    Context context{program_path};
    DeviceConfig config;
    config.extension = make_unique<RemoteDeviceConfigExt>(
        string{host}, port, string{token},
        5'000u, 60'000u, 32u * 1024u * 1024u,
        true, 64u * 1024u, string{local_backend});
    auto device = context.create_device("remote", &config, false);
    expect(device.query("remote.local_present_backend") == local_backend);

    constexpr auto resolution = make_uint2(160u, 96u);
    auto stream = device.create_stream(StreamTag::GRAPHICS);
    Window window{"Remote local swapchain", resolution};
    auto swapchain = device.create_swapchain(
        stream,
        SwapchainOption{
            .display = window.native_display(),
            .window = window.native_handle(),
            .size = resolution,
            .wants_hdr = false,
            .wants_vsync = false,
            .back_buffer_count = 3u});
    auto image = device.create_image<float>(
        swapchain.backend_storage(), resolution);
    Kernel2D render = [resolution](ImageFloat target, Float phase) noexcept {
        auto p = dispatch_id().xy();
        auto uv = make_float2(p) / make_float2(resolution);
        target.write(p, make_float4(
                            uv.x, uv.y, phase, 1.0f));
    };
    auto shader = device.compile(render);
    for (auto frame = 0u; frame < 3u; frame++) {
        stream << shader(image, static_cast<float>(frame + 1u) / 3.0f)
                      .dispatch(resolution)
               << swapchain.present(image)
               << synchronize();
        window.poll_events();
    }

    vector<std::byte> pixels(image.view().size_bytes());
    stream << image.copy_to(span{pixels}) << synchronize();
    expect(std::any_of(
        pixels.begin(), pixels.end(),
        [](std::byte value) noexcept { return value != std::byte{}; }));
    window.set_should_close();
    window.poll_events();
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 5 || argv[1] == nullptr || argv[1][0] == '\0' ||
        argv[2] == nullptr || argv[2][0] == '\0' ||
        argv[3] == nullptr || argv[3][0] == '\0' ||
        argv[4] == nullptr) {
        LUISA_WARNING(
            "Usage: {} <local-present-backend> <host> <port> <token>", argv[0]);
        return 1;
    }
    uint64_t port{};
    auto port_text = string_view{argv[3]};
    auto port_result = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    if (port_result.ec != std::errc{} ||
        port_result.ptr != port_text.data() + port_text.size() ||
        port == 0u || port > std::numeric_limits<uint16_t>::max()) {
        LUISA_WARNING("Invalid remote server port '{}'.", port_text);
        return 1;
    }
    const char *ut_argv[]{argv[0]};
    boost::ut::detail::cfg::parse_arg_with_fallback(1, ut_argv);
    "remote_local_swapchain"_test = [&] {
        test_local_swapchain(
            argv[0], argv[1], argv[2],
            static_cast<uint16_t>(port), argv[4]);
    };
}
