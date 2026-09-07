// Native-backend integration test for the C++ remote backend.
//
// A real backend is hosted by an independently launched server and driven
// exclusively through a remote Device. The test covers AST reconstruction/JIT,
// asynchronous stream completion, buffer and texture I/O, resource bindings,
// timeline events, and cold/hot content-addressed upload paths.

#include "ut/ut.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <system_error>

#include <luisa/backends/ext/remote_config_ext.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] uint64_t parse_u64(string_view text) noexcept {
    uint64_t value{};
    auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    expect(result.ec == std::errc{});
    expect(result.ptr == text.data() + text.size());
    return value;
}

void test_native_backend(const char *program_path, string_view backend,
                         string_view host, uint16_t port,
                         string_view token) {
    Context remote_context{program_path};
    DeviceConfig remote_config;
    remote_config.headless = true;
    remote_config.extension = make_unique<RemoteDeviceConfigExt>(
        string{host}, port, string{token},
        5'000u, 60'000u, 32u * 1024u * 1024u,
        true, 64u * 1024u);
    auto device = remote_context.create_device(
        "remote", &remote_config, false);

    expect(device.query("remote.native_backend") == backend);
    expect(device.query("remote.blob_cache.enabled") == "true");

    constexpr auto element_count = 32u * 1024u;
    constexpr auto bias = 0x31415926u;
    vector<uint> input(element_count);
    vector<uint> output(element_count, 0u);
    for (auto i = 0u; i < element_count; i++) {
        input[i] = (i * 747796405u + 2891336453u) ^ (i >> 3u);
    }

    auto source = device.create_buffer<uint>(element_count);
    auto copied = device.create_buffer<uint>(element_count);
    auto transformed = device.create_buffer<uint>(element_count);
    auto stream = device.create_stream();

    auto initial_misses = parse_u64(
        device.query("remote.blob_cache.misses"));
    auto initial_uploaded_bytes = parse_u64(
        device.query("remote.blob_cache.uploaded_bytes"));
    stream << source.copy_from(span{input}) << synchronize();
    auto cold_misses = parse_u64(
        device.query("remote.blob_cache.misses"));
    auto cold_uploaded_bytes = parse_u64(
        device.query("remote.blob_cache.uploaded_bytes"));
    expect(cold_misses > initial_misses);
    expect(cold_uploaded_bytes ==
           initial_uploaded_bytes + input.size() * sizeof(uint));

    Callable transform = [](UInt value, UInt index, UInt offset) noexcept {
        return (value ^ (index * 1664525u + 1013904223u)) + offset;
    };
    Kernel1D kernel = [&transform](
                          BufferUInt in, BufferUInt out,
                          UInt offset) noexcept {
        set_block_size(64u);
        auto index = dispatch_x();
        out.write(index, transform(in.read(index), index, offset));
    };
    auto shader = device.compile(kernel);

    auto hot_hits = parse_u64(
        device.query("remote.blob_cache.hits"));
    auto hot_uploaded_bytes = parse_u64(
        device.query("remote.blob_cache.uploaded_bytes"));
    auto completion_called = false;
    stream << source.copy_from(span{input})
           << source.copy_to(copied.view())
           << shader(copied, transformed, bias).dispatch(element_count)
           << transformed.copy_to(span{output})
           << [&completion_called] { completion_called = true; }
           << synchronize();
    expect(completion_called);
    expect(parse_u64(device.query("remote.blob_cache.hits")) > hot_hits);
    expect(parse_u64(
               device.query("remote.blob_cache.uploaded_bytes")) ==
           hot_uploaded_bytes);

    auto buffers_match = true;
    for (auto i = 0u; i < element_count; i++) {
        auto expected =
            (input[i] ^ (i * 1664525u + 1013904223u)) + bias;
        if (output[i] != expected) {
            LUISA_WARNING(
                "Remote native buffer mismatch at {}: expected {}, got {}.",
                i, expected, output[i]);
            buffers_match = false;
            break;
        }
    }
    expect(buffers_match);

    constexpr auto bound_count = 257u;
    auto bound_output = device.create_buffer<uint>(bound_count);
    vector<uint> bound_readback(bound_count, 0u);
    auto bound_shader = device.compile<1>([&]() noexcept {
        auto index = dispatch_x();
        bound_output->write(index, index * 17u + 9u);
    });
    stream << bound_shader().dispatch(bound_count)
           << bound_output.copy_to(span{bound_readback})
           << synchronize();
    auto binding_matches = true;
    for (auto i = 0u; i < bound_count; i++) {
        if (bound_readback[i] != i * 17u + 9u) {
            binding_matches = false;
            break;
        }
    }
    expect(binding_matches);

    constexpr auto image_size = make_uint2(32u, 24u);
    auto image = device.create_image<float>(PixelStorage::FLOAT4, image_size);
    vector<float4> image_input(
        static_cast<size_t>(image_size.x) * image_size.y);
    vector<float4> image_output(image_input.size());
    for (auto y = 0u; y < image_size.y; y++) {
        for (auto x = 0u; x < image_size.x; x++) {
            auto index = static_cast<size_t>(y) * image_size.x + x;
            image_input[index] = make_float4(
                static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(x + y), 1.0f);
        }
    }
    Kernel2D image_kernel = [](ImageFloat target) noexcept {
        auto coordinate = dispatch_id().xy();
        auto value = target.read(coordinate);
        target.write(coordinate, make_float4(
                                     value.z + 0.5f,
                                     value.x * 2.0f,
                                     value.y + 1.0f,
                                     value.w));
    };
    auto image_shader = device.compile(image_kernel);
    stream << image.copy_from(span{image_input})
           << image_shader(image).dispatch(image_size)
           << image.copy_to(span{image_output})
           << synchronize();
    auto image_matches = true;
    for (auto i = 0u; i < image_output.size(); i++) {
        auto expected = make_float4(
            image_input[i].z + 0.5f,
            image_input[i].x * 2.0f,
            image_input[i].y + 1.0f,
            image_input[i].w);
        if (any(image_output[i] != expected)) {
            image_matches = false;
            break;
        }
    }
    expect(image_matches);

    auto event = device.create_timeline_event();
    stream << event.signal(5u);
    event.synchronize(5u);
    expect(event.is_completed(5u));
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 5 || argv[1] == nullptr || argv[1][0] == '\0' ||
        argv[2] == nullptr || argv[2][0] == '\0' ||
        argv[3] == nullptr || argv[3][0] == '\0' ||
        argv[4] == nullptr) {
        LUISA_WARNING(
            "Usage: {} <expected-native-backend> <host> <port> <token>",
            argv[0]);
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
    // Host, port, and token are client configuration, not Boost.UT filters.
    // Parse a sanitized argv so the runner executes the test unconditionally.
    const char *ut_argv[]{argv[0]};
    boost::ut::detail::cfg::parse_arg_with_fallback(1, ut_argv);
    "remote_native_backend_e2e"_test = [&] {
        test_native_backend(
            argv[0], argv[1], argv[2],
            static_cast<uint16_t>(port), argv[4]);
    };
}
