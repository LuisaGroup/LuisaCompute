#include "ut/ut.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/binary_io.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

class CapturingBinaryIO final : public BinaryIO {

private:
    mutable size_t _last_shader_cache_size{};

public:
    void clear_shader_cache() const noexcept override {
        _last_shader_cache_size = 0u;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(luisa::string_view) const noexcept override {
        return nullptr;
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view,
        luisa::span<const std::byte> data) const noexcept override {
        _last_shader_cache_size = data.size_bytes();
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    void reset_capture() const noexcept {
        _last_shader_cache_size = 0u;
    }

    [[nodiscard]] size_t last_shader_cache_size() const noexcept {
        return _last_shader_cache_size;
    }
};

static constexpr auto callable_round_count = 192u;

[[nodiscard]] uint32_t scramble_reference(uint32_t value) noexcept {
    for (auto round = 0u; round < callable_round_count; round++) {
        auto right_shift = round % 15u + 1u;
        auto left_shift = round % 13u + 1u;
        auto multiplier = 0x9e3779b1u + round * 0x85ebca6bu;
        auto additive = 0xc2b2ae35u + round * 0x27d4eb2du;
        value = (value ^ (value >> right_shift)) * multiplier;
        value ^= value << left_shift;
        value += additive;
    }
    return value;
}

struct CompileResult {
    size_t artifact_size{};
    luisa::vector<uint32_t> values;
};

[[nodiscard]] CompileResult compile_reused_callable(
    Device &device, CapturingBinaryIO &binary_io,
    uint32_t reuse_count, uint32_t seed) noexcept {
    Callable scramble = [](UInt value) noexcept {
        for (auto round = 0u;
             round < callable_round_count; round++) {
            auto right_shift = round % 15u + 1u;
            auto left_shift = round % 13u + 1u;
            auto multiplier =
                0x9e3779b1u + round * 0x85ebca6bu;
            auto additive =
                0xc2b2ae35u + round * 0x27d4eb2du;
            value = (value ^ (value >> right_shift)) * multiplier;
            value ^= value << left_shift;
            value += additive;
        }
        return value;
    };
    Kernel1D kernel = [&scramble, reuse_count](
                          BufferUInt output,
                          UInt initial_value) noexcept {
        for (auto use = 0u; use < reuse_count; use++) {
            output.write(
                use,
                scramble(initial_value +
                         use * 0x165667b1u));
        }
    };

    binary_io.reset_capture();
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = true});
    auto artifact_size = binary_io.last_shader_cache_size();
    auto output = device.create_buffer<uint32_t>(reuse_count);
    luisa::vector<uint32_t> values(reuse_count);
    auto stream = device.create_stream();
    stream << shader(output, seed).dispatch(1u)
           << output.copy_to(values.data())
           << synchronize();
    return CompileResult{
        .artifact_size = artifact_size,
        .values = std::move(values)};
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    CapturingBinaryIO binary_io;
    Context context{program_path};
    DeviceConfig config{.binary_io = &binary_io};
    auto device = context.create_device("hip", &config);

    "HIP preserves shared DSL callable boundaries after optimization"_test =
        [&] {
            constexpr auto seed = 0x12345678u;
            constexpr auto shared_use_count = 16u;
            auto single = compile_reused_callable(
                device, binary_io, 1u, seed);
            auto shared = compile_reused_callable(
                device, binary_io, shared_use_count, seed);

            expect(single.artifact_size != 0u);
            expect(shared.artifact_size != 0u);
            for (auto use = 0u; use < shared_use_count; use++) {
                auto input = seed + use * 0x165667b1u;
                expect(shared.values[use] ==
                       scramble_reference(input));
            }
            expect(single.values.front() == shared.values.front());

            LUISA_INFO(
                "HIP callable-boundary regression: one use = {} bytes, "
                "{} uses = {} bytes.",
                single.artifact_size, shared_use_count,
                shared.artifact_size);

            // The callable body is intentionally much larger than its call
            // sites. Reusing it must grow the code object by calls and stores,
            // not by sixteen copies of the body. The fixed allowance covers
            // target-dependent code-object metadata without hiding renewed
            // megakernel expansion.
            constexpr auto maximum_reuse_overhead = 96u * 1024u;
            expect(shared.artifact_size <=
                   single.artifact_size + maximum_reuse_overhead)
                << "shared Luisa Callable was repeatedly expanded into the "
                   "HIP kernel";
        };
}
