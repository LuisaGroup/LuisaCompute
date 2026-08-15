#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <optional>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void set_environment_variable(
    const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;

    explicit ScopedEnvironmentVariable(
        const char *env_name, const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(),
            previous ? previous->c_str() : nullptr);
    }
};

[[nodiscard]] bool invalid_worker_override_fails_closed(
    const char *program) noexcept {
#if defined(__unix__) || defined(__APPLE__)
    auto child = fork();
    if (child < 0) { return false; }
    if (child == 0) {
        const rlimit no_core{0u, 0u};
        static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
        set_environment_variable("LUISA_SIMD_WORKER_COUNT", "invalid");
        Context context{program};
        auto device = context.create_device("simd");
        static_cast<void>(device.compute_warp_size());
        _exit(EXIT_SUCCESS);
    }
    auto child_status = 0;
    while (waitpid(child, &child_status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(child_status) &&
           WTERMSIG(child_status) == SIGABRT;
#else
    static_cast<void>(program);
    return true;
#endif
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    expect(invalid_worker_override_fails_closed(
        argc > 0 ? argv[0] : ""))
        << "invalid SIMD worker override must fail closed";
    Context context{argc > 0 ? argv[0] : ""};
    ScopedEnvironmentVariable width_override{
        "LUISA_SIMD_WARP_WIDTH", "2"};
    ScopedEnvironmentVariable worker_override{
        "LUISA_SIMD_WORKER_COUNT", "1"};

    {
        auto device = context.create_device("simd");
        expect(device.compute_warp_size() == 2u)
            << "SIMD environment width override was not honored";
    }

    {
        ScopedEnvironmentVariable invalid_worker_override{
            "LUISA_SIMD_WORKER_COUNT", "invalid"};
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(1u, 1u);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == 1u)
            << "explicit SIMD configuration must override the environment";
    }

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        expect(device.compute_warp_size() == width)
            << "explicit SIMD width must override the environment";

        constexpr auto block_threads = 32u;
        auto output = device.create_buffer<uint>(block_threads);
        Kernel1D kernel = [width](BufferUInt result) noexcept {
            // Block size and logical warp size are separate contracts. Like
            // fallback's scalar thread loop, SIMD partitions this 32-thread
            // block into packets of the configured width.
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto lane = warp_lane_id();
            auto sum = warp_active_sum(lane + 1u);
            result.write(dispatch_x(), sum + lane);
        };
        auto shader = device.compile(kernel);
        auto stream = device.create_stream();
        luisa::vector<uint> host(block_threads, 0u);
        stream << shader(output).dispatch(block_threads)
               << output.copy_to(luisa::span{host})
               << synchronize();

        auto sum = width * (width + 1u) / 2u;
        for (auto thread = 0u; thread < block_threads; thread++) {
            auto lane = thread % width;
            expect(host[thread] == sum + lane)
                << "SIMD thread-block packet partition mismatch";
        }

        // DSL $outline adds source-comment metadata to its call site. That
        // metadata must not prevent the SIMD compiler's mandatory callable
        // legalization pass from inlining the outlined region.
        Kernel1D outlined_kernel = [width](BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            UInt value = dispatch_x();
            $outline {
                value = value * 3u + 7u;
            };
            result.write(dispatch_x(), value);
        };
        auto outlined_shader = device.compile(outlined_kernel);
        stream << outlined_shader(output).dispatch(block_threads)
               << output.copy_to(luisa::span{host})
               << synchronize();
        for (auto thread = 0u; thread < block_threads; thread++) {
            expect(host[thread] == thread * 3u + 7u)
                << "SIMD outlined callable legalization mismatch";
        }

        // Regression: LLVM integer div/rem may lower to trapping scalar
        // instructions even when a zero divisor exists only in an inactive
        // tail lane. Sanitization therefore has to happen before the vector
        // operation; masking the result afterwards is too late.
        auto tail_threads = width == 1u ? 1u : width - 1u;
        auto tail_input = device.create_buffer<uint>(tail_threads);
        auto tail_output = device.create_buffer<uint>(tail_threads);
        Kernel1D tail_kernel = [width](
                                   BufferUInt input,
                                   BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto divisor = input.read(dispatch_x());
            result.write(dispatch_x(), 17u % divisor);
        };
        auto tail_shader = device.compile(tail_kernel);
        luisa::vector<uint> tail_divisors(tail_threads, 3u);
        luisa::vector<uint> tail_host(tail_threads, 0u);
        stream << tail_input.copy_from(luisa::span{tail_divisors})
               << tail_shader(tail_input, tail_output).dispatch(tail_threads)
               << tail_output.copy_to(luisa::span{tail_host})
               << synchronize();
        for (auto value : tail_host) {
            expect(value == 2u)
                << "inactive SIMD tail lane reached integer remainder";
        }

        // Exercise the runtime's fixed-width texture packet callback rather
        // than only the generic Schedule-to-LLVM packet ABI. The first
        // 32-thread row is fully contiguous for every supported width, while
        // the final texel exercises an inactive packet tail.
        constexpr auto image_size = make_uint2(33u, 3u);
        Kernel2D image_kernel = [width](ImageFloat target) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto value = target.read(coordinate);
            target.write(
                coordinate,
                value + make_float4(1.0f, 2.0f, 3.0f, 4.0f));
        };
        auto image_shader = device.compile(image_kernel);
        auto check_float_image = [&](const char *failure_message) noexcept {
            auto image = device.create_image<float>(
                PixelStorage::FLOAT4, image_size);
            luisa::vector<float4> image_input(image_size.x * image_size.y);
            luisa::vector<float4> image_output(image_input.size());
            for (auto y = 0u; y < image_size.y; y++) {
                for (auto x = 0u; x < image_size.x; x++) {
                    image_input[y * image_size.x + x] = make_float4(
                        static_cast<float>(x), static_cast<float>(y),
                        static_cast<float>(x + y), 1.0f);
                }
            }
            stream << image.copy_from(luisa::span{image_input})
                   << image_shader(image).dispatch(image_size)
                   << image.copy_to(luisa::span{image_output})
                   << synchronize();
            for (auto i = size_t{0u}; i < image_input.size(); i++) {
                expect(all(image_output[i] ==
                           image_input[i] +
                               make_float4(1.0f, 2.0f, 3.0f, 4.0f)))
                    << failure_message;
            }
        };
        check_float_image("SIMD contiguous FLOAT4 packet mismatch");
        {
            ScopedEnvironmentVariable disable_contiguous_packets{
                "LUISA_SIMD_DISABLE_CONTIGUOUS_TEXTURE_PACKETS", "1"};
            check_float_image("SIMD generic FLOAT4 packet mismatch");
        }

        auto uint_image = device.create_image<uint>(
            PixelStorage::INT4, image_size);
        luisa::vector<uint4> uint_image_input(image_size.x * image_size.y);
        luisa::vector<uint4> uint_image_output(uint_image_input.size());
        for (auto y = 0u; y < image_size.y; y++) {
            for (auto x = 0u; x < image_size.x; x++) {
                uint_image_input[y * image_size.x + x] = make_uint4(
                    x, y, x + y, x * 3u + y * 5u);
            }
        }
        Kernel2D uint_image_kernel = [width](ImageUInt target) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto coordinate = dispatch_id().xy();
            auto value = target.read(coordinate);
            target.write(
                coordinate,
                value + make_uint4(5u, 7u, 11u, 13u));
        };
        auto uint_image_shader = device.compile(uint_image_kernel);
        stream << uint_image.copy_from(luisa::span{uint_image_input})
               << uint_image_shader(uint_image).dispatch(image_size)
               << uint_image.copy_to(luisa::span{uint_image_output})
               << synchronize();
        for (auto i = size_t{0u}; i < uint_image_input.size(); i++) {
            expect(all(uint_image_output[i] ==
                       uint_image_input[i] +
                           make_uint4(5u, 7u, 11u, 13u)))
                << "SIMD contiguous INT4 packet mismatch";
        }

        // Indirect dispatch uses a backend-owned header/record ABI. Author
        // more records than the capacity to exercise masked out-of-range
        // lanes and an inactive packet tail at every supported width. A zero
        // block dimension deliberately invalidates record 2.
        constexpr auto indirect_capacity = 7u;
        auto indirect = device.create_indirect_dispatch_buffer(
            indirect_capacity);
        auto indirect_output = device.create_buffer<uint>(
            indirect_capacity);
        Kernel1D set_indirect_count = [](
                                          Var<IndirectDispatchBuffer> buffer) noexcept {
            set_block_size(32u, 1u, 1u);
            buffer.set_dispatch_count(indirect_capacity + 5u);
        };
        Kernel1D write_indirect_records = [](
                                              Var<IndirectDispatchBuffer> buffer) noexcept {
            set_block_size(32u, 1u, 1u);
            auto index = dispatch_x();
            auto block_size = ite(
                index == 2u,
                make_uint3(0u, 1u, 1u),
                make_uint3(8u, 1u, 1u));
            buffer.set_kernel(
                index, block_size,
                make_uint3(index + 1u, 1u, 1u), index);
        };
        Kernel1D consume_indirect = [](
                                        BufferUInt result) noexcept {
            set_block_size(32u, 1u, 1u);
            result.atomic(kernel_id()).fetch_add(dispatch_size().x);
        };
        auto set_indirect_count_shader = device.compile(
            set_indirect_count);
        auto write_indirect_records_shader = device.compile(
            write_indirect_records);
        auto consume_indirect_shader = device.compile(
            consume_indirect);
        luisa::vector<uint> indirect_host(
            indirect_capacity, 0u);
        stream << indirect_output.copy_from(
                      luisa::span{indirect_host})
               << set_indirect_count_shader(indirect).dispatch(width)
               << write_indirect_records_shader(indirect).dispatch(
                      indirect_capacity + 2u)
               << consume_indirect_shader(indirect_output).dispatch(indirect, 1u, 4u)
               << consume_indirect_shader(indirect_output).dispatch(indirect, 6u)
               << consume_indirect_shader(indirect_output).dispatch(indirect)
               << indirect_output.copy_to(
                      luisa::span{indirect_host})
               << synchronize();
        constexpr std::array indirect_expected{
            1u, 8u, 0u, 32u, 50u, 36u, 98u};
        for (auto i = 0u; i < indirect_capacity; i++) {
            expect(indirect_host[i] == indirect_expected[i])
                << "SIMD indirect dispatch mismatch";
        }

        // Batched direct dispatch shares kernel_id() semantics with indirect
        // records: each logical command receives its zero-based batch index.
        std::fill(
            indirect_host.begin(), indirect_host.end(), 0u);
        constexpr std::array direct_batches{
            make_uint3(1u, 1u, 1u),
            make_uint3(2u, 1u, 1u),
            make_uint3(3u, 1u, 1u)};
        stream << indirect_output.copy_from(
                      luisa::span{indirect_host})
               << consume_indirect_shader(indirect_output).dispatch(luisa::span{direct_batches})
               << indirect_output.copy_to(
                      luisa::span{indirect_host})
               << synchronize();
        constexpr std::array direct_expected{
            1u, 4u, 9u, 0u, 0u, 0u, 0u};
        for (auto i = 0u; i < indirect_capacity; i++) {
            expect(indirect_host[i] == direct_expected[i])
                << "SIMD batched dispatch kernel-id mismatch";
        }
    }
}
