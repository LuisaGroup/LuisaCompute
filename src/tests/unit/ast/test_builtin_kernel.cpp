// Test for the runtime builtin fill kernels.
// This test covers buffer, image, and volume fills through the public BuiltinKernel API.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/runtime/builtin_kernel.h>

#include <algorithm>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct MyStruct {
    uint a;
    uint b;
    uint c;
    uint d;
};

template<typename T>
[[nodiscard]] bool all_equal(const std::vector<T> &values, const T &expected) noexcept {
    return std::all_of(values.cbegin(), values.cend(), [&](const T &value) noexcept {
        return value == expected;
    });
}

template<typename T>
[[nodiscard]] bool all_texels_equal(const std::vector<Vector<T, 4>> &values, T expected) noexcept {
    return std::all_of(values.cbegin(), values.cend(), [&](const Vector<T, 4> &value) noexcept {
        return value.x == expected && value.y == expected &&
               value.z == expected && value.w == expected;
    });
}

void test_builtin_kernel(Device &device) {
    log_level_verbose();

    auto stream = device.create_stream();
    BuiltinKernel builtin{device};
    builtin.compile_all(device);

    {
        constexpr size_t buffer_size = 1024u;
        auto buffer = device.create_buffer<uint>(buffer_size);
        CommandList commands;
        builtin.fill_buffer(commands, buffer.view(), 42u);
        stream << commands.commit() << synchronize();

        std::vector<uint> result(buffer_size);
        stream << buffer.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, 42u)) << "builtin uint buffer fill must write every element";
    }

    {
        constexpr size_t buffer_size = 256u;
        auto buffer = device.create_buffer<MyStruct>(buffer_size);
        constexpr MyStruct expected{1u, 2u, 3u, 4u};
        CommandList commands;
        builtin.fill_buffer(commands, buffer.view(), expected);
        stream << commands.commit() << synchronize();

        std::vector<MyStruct> result(buffer_size);
        stream << buffer.copy_to(luisa::span{result}) << synchronize();
        auto success = std::all_of(result.cbegin(), result.cend(), [expected](const MyStruct &value) noexcept {
            return value.a == expected.a && value.b == expected.b &&
                   value.c == expected.c && value.d == expected.d;
        });
        expect(success) << "builtin structured buffer fill must copy every field";
    }

    {
        constexpr uint2 size{8u, 7u};
        auto image = device.create_image<uint>(PixelStorage::INT4, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), 255u);
        stream << commands.commit() << synchronize();

        std::vector<uint4> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, 255u)) << "builtin uint image fill must splat to all channels";
    }

    {
        constexpr uint2 size{8u, 7u};
        auto image = device.create_image<int>(PixelStorage::INT4, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), -42);
        stream << commands.commit() << synchronize();

        std::vector<int4> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, -42)) << "builtin int image fill must splat to all channels";
    }

    {
        constexpr uint2 size{8u, 7u};
        auto image = device.create_image<float>(PixelStorage::FLOAT4, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), 2.718f);
        stream << commands.commit() << synchronize();

        std::vector<float4> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, 2.718f)) << "builtin float image fill must splat to all channels";
    }

    {
        constexpr uint3 size{5u, 4u, 3u};
        auto volume = device.create_volume<uint>(PixelStorage::INT4, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), 100u);
        stream << commands.commit() << synchronize();

        std::vector<uint4> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, 100u)) << "builtin uint volume fill must splat to all channels";
    }

    {
        constexpr uint3 size{5u, 4u, 3u};
        auto volume = device.create_volume<int>(PixelStorage::INT4, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), -50);
        stream << commands.commit() << synchronize();

        std::vector<int4> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, -50)) << "builtin int volume fill must splat to all channels";
    }

    {
        constexpr uint3 size{5u, 4u, 3u};
        auto volume = device.create_volume<float>(PixelStorage::FLOAT4, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), 3.14f);
        stream << commands.commit() << synchronize();

        std::vector<float4> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_texels_equal(result, 3.14f)) << "builtin float volume fill must splat to all channels";
    }

    {
        constexpr uint2 size{7u, 5u};
        auto image = device.create_image<uint>(PixelStorage::INT1, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), 123u);
        stream << commands.commit() << synchronize();

        std::vector<uint> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, 123u)) << "builtin uint image fill must support single-channel storage";
    }

    {
        constexpr uint2 size{7u, 5u};
        auto image = device.create_image<int>(PixelStorage::INT1, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), -17);
        stream << commands.commit() << synchronize();

        std::vector<int> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, -17)) << "builtin int image fill must support single-channel storage";
    }

    {
        constexpr uint2 size{7u, 5u};
        auto image = device.create_image<float>(PixelStorage::FLOAT1, size);
        CommandList commands;
        builtin.fill_image(commands, image.view(), 1.25f);
        stream << commands.commit() << synchronize();

        std::vector<float> result(static_cast<size_t>(size.x) * size.y);
        stream << image.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, 1.25f)) << "builtin float image fill must support single-channel storage";
    }

    {
        constexpr uint3 size{4u, 3u, 2u};
        auto volume = device.create_volume<uint>(PixelStorage::INT1, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), 77u);
        stream << commands.commit() << synchronize();

        std::vector<uint> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, 77u)) << "builtin uint volume fill must support single-channel storage";
    }

    {
        constexpr uint3 size{4u, 3u, 2u};
        auto volume = device.create_volume<int>(PixelStorage::INT1, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), -99);
        stream << commands.commit() << synchronize();

        std::vector<int> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, -99)) << "builtin int volume fill must support single-channel storage";
    }

    {
        constexpr uint3 size{4u, 3u, 2u};
        auto volume = device.create_volume<float>(PixelStorage::FLOAT1, size);
        CommandList commands;
        builtin.fill_volume(commands, volume.view(), 1.414f);
        stream << commands.commit() << synchronize();

        std::vector<float> result(static_cast<size_t>(size.x) * size.y * size.z);
        stream << volume.copy_to(luisa::span{result}) << synchronize();
        expect(all_equal(result, 1.414f)) << "builtin float volume fill must support single-channel storage";
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    std::vector<const char *> ut_args;
    ut_args.reserve(static_cast<size_t>(argc - 1));
    ut_args.emplace_back(argv[0]);
    for (auto i = 2; i < argc; i++) {
        ut_args.emplace_back(argv[i]);
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(static_cast<int>(ut_args.size()), ut_args.data());
    "builtin_kernel_fill_roundtrip"_test = [&] {
        test_builtin_kernel(dc->device);
    };
}
