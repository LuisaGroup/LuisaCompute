#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>

#include <luisa/core/binary_io.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

class SourceTrackingBinaryIO final : public BinaryIO {

public:
    mutable size_t shader_source_write_count{};
    mutable std::string last_shader_source;

    void clear_shader_cache() const noexcept override {}

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
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_source(
        luisa::string_view,
        luisa::span<const std::byte> source) const noexcept override {
        shader_source_write_count++;
        last_shader_source.assign(
            reinterpret_cast<const char *>(source.data()), source.size());
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }
};

void set_dump_source_environment(const char *value) noexcept {
#ifdef _WIN32
    _putenv_s("LUISA_DUMP_SOURCE", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("LUISA_DUMP_SOURCE");
    } else {
        setenv("LUISA_DUMP_SOURCE", value, 1);
    }
#endif
}

[[nodiscard]] bool close(float4 a, float4 b) noexcept {
    auto d = abs(a - b);
    return all(d < make_float4(2.0e-2f));
}

void test_metal_codegen_regressions(
    Device &device, SourceTrackingBinaryIO &binary_io) {
    constexpr auto size = make_uint2(2u, 2u);
    auto image = device.create_image<float>(PixelStorage::BYTE4, size);
    std::array<uint8_t, 16u> pixels{
        255u, 0u, 0u, 255u, 0u, 255u, 0u, 255u,
        0u, 0u, 255u, 255u, 255u, 255u, 255u, 255u};
    auto output = device.create_buffer<float4>(2u);
    std::array<float4, 2u> result{};

    auto mutate = Callable<void(float2 &)>{[](Var<float2> &v) noexcept {
        v = v + make_float2(1.0f);
    }};
    Kernel1D kernel = [&](ImageFloat tex, BufferFloat4 out) noexcept {
        auto id = dispatch_id().x;
        Float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        auto &builder = *luisa::compute::detail::FunctionBuilder::current();
        auto swizzle = luisa::compute::detail::Ref<float2>{
            builder.swizzle(Type::of<float2>(), value.expression(), 2u, 0x01u)};
        mutate(swizzle);
        auto sampled = builder.call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
            {tex.expression(), builder.literal(Type::of<float2>(), make_float2(0.25f, 0.25f)),
             builder.literal(Type::of<uint32_t>(), static_cast<uint32_t>(Sampler::Filter::POINT)),
             builder.literal(Type::of<uint32_t>(), static_cast<uint32_t>(Sampler::Address::EDGE))});
        out.write(id, value + def<float4>(sampled));
    };
    auto shader = device.compile(kernel);
    auto stream = device.create_stream();
    stream << image.copy_from(luisa::span{pixels})
           << shader(image, output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    expect(close(result[0], make_float4(3.0f, 3.0f, 3.0f, 5.0f)))
        << "Metal sampled textures and mutable swizzle references must compile and execute";

    auto mutate_and_return = Callable<float(float2 &)>{[](Var<float2> &v) noexcept {
        v = v + make_float2(1.0f);
        return v.x;
    }};
    Kernel1D overlapping_writeback = [&](BufferFloat4 out) noexcept {
        Float4 value = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        auto &builder = *luisa::compute::detail::FunctionBuilder::current();
        auto swizzle = luisa::compute::detail::Ref<float2>{
            builder.swizzle(Type::of<float2>(), value.expression(), 2u, 0x01u)};
        value.x = mutate_and_return(swizzle);
        out.write(0u, value);
    };
    auto overlapping_writeback_shader = device.compile(overlapping_writeback);
    stream << overlapping_writeback_shader(output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    expect(close(result[0], make_float4(3.0f, 3.0f, 3.0f, 4.0f)))
        << "Mutable swizzle writeback must precede assignment to an overlapping lvalue";

    Kernel1D combined = [&](ImageFloat tex, BufferFloat4 out) noexcept {
        auto &builder = *luisa::compute::detail::FunctionBuilder::current();
        auto sampled = builder.call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
            {tex.expression(), builder.literal(Type::of<float2>(), make_float2(0.25f, 0.25f)),
             builder.literal(Type::of<uint32_t>(), static_cast<uint32_t>(Sampler::Filter::POINT)),
             builder.literal(Type::of<uint32_t>(), static_cast<uint32_t>(Sampler::Address::EDGE))});
        auto sampled_value = def<float4>(sampled);
        out.write(0u, sampled_value);
        tex.write(make_uint2(0u), sampled_value);
    };
    auto combined_shader = device.compile(combined);
    stream << combined_shader(image, output).dispatch(1u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    expect(close(result[0], make_float4(1.0f, 0.0f, 0.0f, 1.0f)))
        << "A sampled and storage texture must use separate Metal access views";

    auto bindless_input = device.create_buffer<uint>(2u);
    auto bindless_output = device.create_buffer<uint>(4u);
    auto bindless = device.create_bindless_array(4u);
    bindless.emplace_on_update(1u, bindless_input);
    bindless.emplace_on_update(2u, bindless_output);
    std::array<uint, 2u> bindless_values{17u, 29u};
    std::array<uint, 4u> bindless_result{};
    Kernel1D uniform_bindless = [](BindlessVar heap) noexcept {
        auto id = dispatch_id().x;
        auto untyped = heap.buffer<uint>(1u, false, true).read(id);
        auto typed = heap.buffer<uint>(1u, true, true).read(id);
        heap.buffer<uint>(2u, false, true).write(id, untyped + typed);
        heap.buffer<uint>(2u, true, true).write(id + 2u, untyped + typed + 1u);
    };
    auto uniform_bindless_shader = device.compile(uniform_bindless);
    stream << bindless_input.copy_from(luisa::span{bindless_values})
           << bindless.update()
           << uniform_bindless_shader(bindless).dispatch(2u)
           << bindless_output.copy_to(luisa::span{bindless_result})
           << synchronize();
    expect(bindless_result == std::array<uint, 4u>{34u, 58u, 35u, 59u})
        << "Metal uniform and typed-uniform bindless buffers must compile and execute";

    if (device.backend_name() == "metal") {
        auto add_one = Callable<float(float)>{[](Float value) noexcept {
            return value + 1.0f;
        }};
        auto double_value = Callable<float(float)>{[](Float value) noexcept {
            return value * 2.0f;
        }};
        auto make_ordered_kernel = [&](bool reverse_insertion) noexcept {
            return Kernel1D{[&, reverse_insertion](BufferFloat values) noexcept {
                auto *builder = luisa::compute::detail::FunctionBuilder::current();
                if (reverse_insertion) {
                    static_cast<void>(
                        builder->func_ref(double_value.function()));
                    static_cast<void>(builder->func_ref(add_one.function()));
                } else {
                    static_cast<void>(builder->func_ref(add_one.function()));
                    static_cast<void>(
                        builder->func_ref(double_value.function()));
                }
                values.write(
                    0u, add_one(3.0f) + double_value(5.0f));
            }};
        };
        auto forward = make_ordered_kernel(false);
        auto reverse = make_ordered_kernel(true);
        expect(forward.function()->function().hash() ==
               reverse.function()->function().hash())
            << "Callable insertion order must not change kernel structure";

        const auto source_writes_before =
            binary_io.shader_source_write_count;
        const auto *old_dump_source = std::getenv("LUISA_DUMP_SOURCE");
        auto old_dump_source_value = old_dump_source == nullptr
                                         ? std::string{}
                                         : std::string{old_dump_source};
        set_dump_source_environment("1");
        auto forward_shader = device.compile(forward);
        const auto source_writes_after_forward =
            binary_io.shader_source_write_count;
        auto reverse_shader = device.compile(reverse);
        set_dump_source_environment(
            old_dump_source == nullptr ? nullptr :
                                         old_dump_source_value.c_str());
        expect(source_writes_after_forward == source_writes_before + 1u)
            << "The first deterministic-order kernel must reach Metal source generation";
        expect(binary_io.shader_source_write_count ==
               source_writes_after_forward)
            << "Equivalent callable graphs with different insertion order must reuse the in-memory Metal shader cache";
        static_cast<void>(forward_shader);
        static_cast<void>(reverse_shader);

        uint root_local_uid = ~0u;
        uint branch_local_uid = ~0u;
        Kernel1D scoped_locals = [&](BufferUInt values) noexcept {
            auto id = dispatch_x();
            UInt result = id + 3u;
            root_local_uid = static_cast<const RefExpr *>(
                                 result.expression())->variable().uid();
            $if (id == 0u) {
                UInt branch_value = result * 2u;
                branch_local_uid = static_cast<const RefExpr *>(
                                       branch_value.expression())->variable().uid();
                result = branch_value;
            };
            values.write(id, result);
        };
        auto scoped_output = device.create_buffer<uint>(2u);
        std::array<uint, 2u> scoped_result{};
        const auto *old_scoped_dump_source = std::getenv("LUISA_DUMP_SOURCE");
        auto old_scoped_dump_source_value =
            old_scoped_dump_source == nullptr ?
                std::string{} : std::string{old_scoped_dump_source};
        set_dump_source_environment("1");
        binary_io.last_shader_source.clear();
        auto scoped_shader = device.compile(scoped_locals);
        set_dump_source_environment(
            old_scoped_dump_source == nullptr ? nullptr :
                                                old_scoped_dump_source_value.c_str());
        stream << scoped_shader(scoped_output).dispatch(2u)
               << scoped_output.copy_to(luisa::span{scoped_result})
               << synchronize();
        expect(scoped_result == std::array<uint, 2u>{6u, 4u})
            << "Lexically scoped Metal locals must preserve execution semantics";
        const auto root_initializer =
            luisa::format("uint v{} = ", root_local_uid);
        const auto branch_initializer =
            luisa::format("uint v{} = ", branch_local_uid);
        expect(binary_io.last_shader_source.find(root_initializer) !=
               std::string::npos)
            << "A root local's first dominating assignment must become its declaration";
        expect(binary_io.last_shader_source.find(branch_initializer) !=
               std::string::npos)
            << "A branch-local first assignment must become a branch-local declaration";
        expect(binary_io.last_shader_source.find(
                   luisa::format("uint v{}{{}};", root_local_uid)) ==
               std::string::npos)
            << "A directly initialized root local must not retain an eager zero declaration";
        expect(binary_io.last_shader_source.find(
                   luisa::format("uint v{}{{}};", branch_local_uid)) ==
               std::string::npos)
            << "A directly initialized branch local must not retain an eager zero declaration";
    }
}

}// namespace

int main(int argc, char *argv[]) {
    SourceTrackingBinaryIO binary_io;
    DeviceConfig config{.binary_io = &binary_io};
    auto dc = luisa::test::create_device_from_ut(argc, argv, &config);
    if (!dc) { return 0; }
    test_metal_codegen_regressions(dc->device, binary_io);
}
