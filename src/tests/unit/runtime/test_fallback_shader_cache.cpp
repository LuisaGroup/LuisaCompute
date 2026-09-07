#include "ut/ut.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <luisa/core/binary_io.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

class MemoryBinaryStream final : public BinaryStream {

private:
    luisa::vector<std::byte> _data;
    size_t _position{};

public:
    explicit MemoryBinaryStream(
        luisa::span<const std::byte> data) noexcept
        : _data{data.begin(), data.end()} {}

    [[nodiscard]] size_t length() const noexcept override {
        return _data.size();
    }

    [[nodiscard]] size_t pos() const noexcept override {
        return _position;
    }

    void read(luisa::span<std::byte> destination) noexcept override {
        if (_position > _data.size() ||
            destination.size() > _data.size() - _position) {
            std::memset(
                destination.data(), 0, destination.size());
            _position = _data.size();
            return;
        }
        std::memcpy(
            destination.data(), _data.data() + _position,
            destination.size());
        _position += destination.size();
    }
};

class MemoryBinaryIO final : public BinaryIO {

private:
    mutable std::unordered_map<
        std::string, std::vector<std::byte>>
        _cache;

public:
    mutable size_t cache_read_count{};
    mutable size_t cache_write_count{};

    void clear_shader_cache() const noexcept override {
        _cache.clear();
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(
        luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(
        luisa::string_view name) const noexcept override {
        cache_read_count++;
        auto iterator = _cache.find(std::string{name});
        if (iterator == _cache.end()) { return nullptr; }
        auto &&data = iterator->second;
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{
                data.data(), data.size()});
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(
        luisa::string_view) const noexcept override {
        return nullptr;
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept override {
        cache_write_count++;
        _cache[std::string{name}] = {
            data.begin(), data.end()};
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }
};

[[nodiscard]] int run_cached_kernel(
    const char *program_path, const BinaryIO *binary_io,
    int value, bool enable_cache) noexcept {
    Context context{program_path};
    DeviceConfig config{.binary_io = binary_io};
    auto device = context.create_device("fallback", &config);
    auto output = device.create_buffer<int>(1u);
    Kernel1D kernel = [](
                          BufferVar<int> result,
                          Int parameter) noexcept {
        result->write(0u, parameter * 3 + 1);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = enable_cache});
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, value).dispatch(1u)
           << output.copy_to(luisa::span{&result, 1})
           << synchronize();
    return result;
}

[[nodiscard]] std::array<uint4, 4u>
run_boolean_comparison_kernel(const char *program_path) noexcept {
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto output = device.create_buffer<uint4>(4u);
    Kernel1D kernel = [](BufferUInt4 result) noexcept {
        const auto index = dispatch_x();
        const auto lhs = (index & 1u) != 0u;
        const auto rhs = index >= 2u;
        const auto scalar_equal = lhs == rhs;
        const auto scalar_not_equal = lhs != rhs;
        const auto vector_equal =
            make_bool2(lhs, !lhs) == make_bool2(rhs, !rhs);
        const auto vector_not_equal =
            make_bool2(lhs, !lhs) != make_bool2(rhs, !rhs);
        const auto pack = [](Bool2 value) noexcept {
            return select(0u, 1u, value.x) |
                   (select(0u, 1u, value.y) << 1u);
        };
        result->write(index,
                      make_uint4(select(0u, 1u, scalar_equal),
                                 select(0u, 1u, scalar_not_equal),
                                 pack(vector_equal),
                                 pack(vector_not_equal)));
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<uint4, 4u> result{};
    stream << shader(output).dispatch(4u)
           << output.copy_to(result.data())
           << synchronize();
    return result;
}

[[nodiscard]] std::array<uint, 4u>
run_assume_kernel(const char *program_path) noexcept {
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto output = device.create_buffer<uint>(4u);
    Kernel1D kernel = [](BufferUInt result) noexcept {
        const auto index = dispatch_x();
        assume(index < 4u);
        result->write(index, index + 1u);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<uint, 4u> result{};
    stream << shader(output).dispatch(4u)
           << output.copy_to(result.data())
           << synchronize();
    return result;
}

[[nodiscard]] std::array<float4, 4u>
run_minimal_codegen_vector_kernel(const char *program_path) noexcept {
    constexpr auto lane_count = 1024u;
    Context context{program_path};
    auto device = context.create_device("fallback");
    auto input = device.create_buffer<float>(4u * lane_count);
    auto output = device.create_buffer<float4>(4u);
    Kernel1D kernel = [](BufferFloat values, BufferFloat4 result) noexcept {
        // Keep scalar SSA values live until they are assembled into vectors.
        // This is the reduced form of the large material-dispatch kernel that
        // exposed FastISel folding four-byte-aligned spills into aligned
        // vector loads. The forced optimization limit below exercises the
        // same O0-IR/O1-machine policy without a production-sized shader.
        constexpr auto kernel_lane_count = 1024u;
        const auto index = dispatch_x();
        luisa::vector<Float> lanes;
        lanes.reserve(kernel_lane_count);
        for (auto lane = 0u; lane < kernel_lane_count; ++lane) {
            lanes.emplace_back(values.read(index * kernel_lane_count + lane));
        }
        Float4 sum = make_float4(0.0f);
        for (auto group = 0u; group < kernel_lane_count / 4u; ++group) {
            const auto lane = group * 4u;
            sum += make_float4(lanes[lane], lanes[lane + 1u],
                               lanes[lane + 2u], lanes[lane + 3u]);
        }
        result->write(index, sum);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false});
    auto stream = device.create_stream();
    std::array<float, 4u * lane_count> values{};
    for (auto index = 0u; index < 4u; ++index) {
        for (auto lane = 0u; lane < lane_count; ++lane) {
            values[index * lane_count + lane] =
                static_cast<float>(index * 1000u + lane);
        }
    }
    std::array<float4, 4u> result{};
    stream << input.copy_from(values.data())
           << shader(input, output).dispatch(4u)
           << output.copy_to(result.data())
           << synchronize();
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    // This must be set before the first fallback backend module is loaded.
#if defined(_WIN32)
    _putenv_s("LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT", "1");
#else
    setenv("LUISA_FALLBACK_OPTIMIZATION_INSTRUCTION_LIMIT", "1", 1);
#endif
    MemoryBinaryIO binary_io;

    "fallback object cache reuses code across devices and keeps uniforms dynamic"_test =
        [&] {
            auto cold_result = run_cached_kernel(
                program_path, &binary_io, 7, true);
            expect(cold_result == 22);
            expect(binary_io.cache_write_count == 2u);
            auto cold_read_count = binary_io.cache_read_count;
            auto cold_write_count = binary_io.cache_write_count;

            auto hot_result = run_cached_kernel(
                program_path, &binary_io, 11, true);
            expect(hot_result == 34);
            expect(
                binary_io.cache_read_count ==
                cold_read_count + 2u);
            expect(
                binary_io.cache_write_count ==
                cold_write_count);

            auto reads_before_disabled =
                binary_io.cache_read_count;
            auto writes_before_disabled =
                binary_io.cache_write_count;
            auto uncached_result = run_cached_kernel(
                program_path, &binary_io, 13, false);
            expect(uncached_result == 40);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled);
        };

    "fallback lowers scalar and vector boolean equality"_test = [&] {
        const auto actual = run_boolean_comparison_kernel(program_path);
        constexpr std::array expected{
            make_uint4(1u, 0u, 3u, 0u),
            make_uint4(0u, 1u, 0u, 3u),
            make_uint4(0u, 1u, 0u, 3u),
            make_uint4(1u, 0u, 3u, 0u)};
        expect(std::memcmp(actual.data(), expected.data(), sizeof(expected)) == 0);
    };

    "fallback lowers scalar boolean assumptions to LLVM i1"_test = [&] {
        const auto actual = run_assume_kernel(program_path);
        constexpr std::array expected{1u, 2u, 3u, 4u};
        expect(actual == expected);
    };

    "fallback minimal codegen preserves vector spill alignment"_test = [&] {
        const auto actual = run_minimal_codegen_vector_kernel(program_path);
        constexpr std::array expected{
            make_float4(130560.0f, 130816.0f, 131072.0f, 131328.0f),
            make_float4(386560.0f, 386816.0f, 387072.0f, 387328.0f),
            make_float4(642560.0f, 642816.0f, 643072.0f, 643328.0f),
            make_float4(898560.0f, 898816.0f, 899072.0f, 899328.0f)};
        expect(std::memcmp(actual.data(), expected.data(), sizeof(expected)) == 0);
    };
}
