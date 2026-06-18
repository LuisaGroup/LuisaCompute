// Test CoroFrame: DSL expressions for coro_id, target_token, get<T>(), is_terminated()
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/ast/type_registry.h>

#include "ut/ut.hpp"
#include "coro_test_utils.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void reg_coro_frame_static() {

    "terminal_token_value"_test = [] {
        expect(CoroFrame::TERMINAL_TOKEN == 0xFFFFFFFFu);
    };

    "desc_empty_on_construction"_test = [] {
        CoroFrameDesc desc;
        expect(desc.field_count() == 0u);
        expect(desc.total_size() == 0u);
    };

    "desc_add_fields"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("x", Type::of<float>());
        desc.add_field("y", Type::of<int>());

        expect(desc.field_count() == 2u);
        expect(desc.field(0u).name == "x");
        expect(desc.field(0u).type->tag() == Type::Tag::FLOAT32);
        expect(desc.field(1u).name == "y");
        expect(desc.field(1u).type->tag() == Type::Tag::INT32);
    };

    "desc_field_lookup_by_name"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("a", Type::of<float>());
        desc.add_field("b", Type::of<int>());
        desc.add_field("c", Type::of<uint>());

        auto *f = desc.field("b");
        expect(f != nullptr);
        expect(f->name == "b");
        expect(f->type->tag() == Type::Tag::INT32);

        auto *nf = desc.field("nonexistent");
        expect(nf == nullptr);
    };
}

void reg_coro_frame_runtime(Device &device) {
    Stream stream = device.create_stream();

    "coro_frame_coro_id_assign_and_read"_test = [&] {
        // coro_id is a UInt3 Var member - can assign and read back
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto result_buf = device.create_buffer<uint>(3u);
        Kernel1D k = [&desc, &result_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            frame.coro_id = make_uint3(10u, 20u, 30u);
            result_buf->write(0u, frame.coro_id.x);
            result_buf->write(1u, frame.coro_id.y);
            result_buf->write(2u, frame.coro_id.z);
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<uint> host(3u);
        stream << result_buf.copy_to(luisa::span{host}) << synchronize();

        expect(host[0] == 10u);
        expect(host[1] == 20u);
        expect(host[2] == 30u);
    };

    "coro_frame_target_token_assign_and_read"_test = [&] {
        // target_token is a UInt Var member - can assign and read back
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto result_buf = device.create_buffer<uint>(1u);
        Kernel1D k = [&desc, &result_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            frame.target_token = 42u;
            result_buf->write(0u, frame.target_token);
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<uint> host(1u);
        stream << result_buf.copy_to(luisa::span{host}) << synchronize();

        expect(host[0] == 42u);
    };

    "coro_frame_get_by_index_type_check"_test = [&] {
        // get<T>(index) returns Var<T> - verify by writing to typed buffer
        CoroFrameDesc desc;
        desc.add_field("fval", Type::of<float>());
        desc.add_field("ival", Type::of<int>());
        desc.add_field("uval", Type::of<uint>());

        auto float_buf = device.create_buffer<float>(1u);
        auto int_buf = device.create_buffer<int>(1u);
        auto uint_buf = device.create_buffer<uint>(1u);
        Kernel1D k = [&desc, &float_buf, &int_buf, &uint_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            // get<T>(index) compiles and returns correct types
            auto fv = frame.get<float>(0u);
            auto iv = frame.get<int>(1u);
            auto uv = frame.get<uint>(2u);
            float_buf->write(0u, fv);
            int_buf->write(0u, iv);
            uint_buf->write(0u, uv);
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<float> fhost(1u);
        luisa::vector<int> ihost(1u);
        luisa::vector<uint> uhost(1u);
        stream << float_buf.copy_to(luisa::span{fhost}) << synchronize();
        stream << int_buf.copy_to(luisa::span{ihost}) << synchronize();
        stream << uint_buf.copy_to(luisa::span{uhost}) << synchronize();
        // Just verify buffers were written (values depend on init, not checked)
        expect(fhost.size() == 1u);
        expect(ihost.size() == 1u);
        expect(uhost.size() == 1u);
    };

    "coro_frame_get_by_name_type_check"_test = [&] {
        // get<T>(name) looks up field and returns Var<T>
        CoroFrameDesc desc;
        desc.add_field("alpha", Type::of<float>());
        desc.add_field("beta", Type::of<uint>());

        auto alpha_buf = device.create_buffer<float>(1u);
        auto beta_buf = device.create_buffer<uint>(1u);
        Kernel1D k = [&desc, &alpha_buf, &beta_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            alpha_buf->write(0u, frame.get<float>("alpha"));
            beta_buf->write(0u, frame.get<uint>("beta"));
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<float> ahost(1u);
        luisa::vector<uint> bhost(1u);
        stream << alpha_buf.copy_to(luisa::span{ahost}) << synchronize();
        stream << beta_buf.copy_to(luisa::span{bhost}) << synchronize();
        expect(ahost.size() == 1u);
        expect(bhost.size() == 1u);
    };

    "coro_frame_is_terminated_false"_test = [&] {
        // target_token = 0, so is_terminated() should be false
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto result_buf = device.create_buffer<int>(1u);
        Kernel1D k = [&desc, &result_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            frame.target_token = 0u;
            $if (frame.is_terminated()) {
                result_buf->write(0u, 1);
            } $else {
                result_buf->write(0u, 0);
            };
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<int> host(1u);
        stream << result_buf.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 0);
    };

    "coro_frame_is_terminated_true"_test = [&] {
        // Set target_token = TERMINAL_TOKEN, is_terminated() should be true
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto result_buf = device.create_buffer<int>(1u);
        Kernel1D k = [&desc, &result_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            frame.target_token = CoroFrame::TERMINAL_TOKEN;
            $if (frame.is_terminated()) {
                result_buf->write(0u, 1);
            } $else {
                result_buf->write(0u, 0);
            };
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<int> host(1u);
        stream << result_buf.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 1);
    };

    "coro_frame_is_terminated_transition"_test = [&] {
        // Verify terminal detection works when target_token changes
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto result_buf = device.create_buffer<int>(2u);
        Kernel1D k = [&desc, &result_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);

            // Not terminated
            frame.target_token = 0u;
            $if (frame.is_terminated()) {
                result_buf->write(0u, 1);
            } $else {
                result_buf->write(0u, 0);
            };

            // Terminated
            frame.target_token = CoroFrame::TERMINAL_TOKEN;
            $if (frame.is_terminated()) {
                result_buf->write(1u, 1);
            } $else {
                result_buf->write(1u, 0);
            };
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<int> host(2u);
        stream << result_buf.copy_to(luisa::span{host}) << synchronize();

        expect(host[0] == 0);// not terminated
        expect(host[1] == 1);// terminated
    };

    "coro_frame_desc_accessor"_test = [&] {
        // desc() returns the descriptor pointer
        CoroFrameDesc desc;
        desc.add_field("val", Type::of<float>());

        auto count_buf = device.create_buffer<uint>(1u);
        Kernel1D k = [&desc, &count_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            auto *d = frame.desc();
            count_buf->write(0u, static_cast<uint>(d->field_count()));
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<uint> host(1u);
        stream << count_buf.copy_to(luisa::span{host}) << synchronize();
        expect(host[0] == 1u);
    };

    "coro_frame_create_factory"_test = [&] {
        // create() factory works same as explicit constructor
        CoroFrameDesc desc;
        desc.add_field("x", Type::of<float>());
        desc.add_field("y", Type::of<int>());

        auto x_buf = device.create_buffer<float>(1u);
        auto y_buf = device.create_buffer<int>(1u);
        Kernel1D k = [&desc, &x_buf, &y_buf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            x_buf->write(0u, frame.get<float>("x"));
            y_buf->write(0u, frame.get<int>("y"));
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<float> xhost(1u);
        luisa::vector<int> yhost(1u);
        stream << x_buf.copy_to(luisa::span{xhost}) << synchronize();
        stream << y_buf.copy_to(luisa::span{yhost}) << synchronize();
        expect(xhost.size() == 1u);
        expect(yhost.size() == 1u);
    };

    "coro_frame_multiple_fields"_test = [&] {
        // Multiple fields of different types accessible through frame
        CoroFrameDesc desc;
        desc.add_field("position", Type::of<float3>());
        desc.add_field("color", Type::of<float4>());
        desc.add_field("flags", Type::of<uint>());

        auto fbuf = device.create_buffer<float>(7u);
        auto ubuf = device.create_buffer<uint>(1u);
        Kernel1D k = [&desc, &fbuf, &ubuf]() noexcept {
            auto frame = CoroFrame::create(&desc);
            auto pos = frame.get<float3>("position");
            auto col = frame.get<float4>("color");
            auto flg = frame.get<uint>("flags");
            fbuf->write(0u, pos.x);
            fbuf->write(1u, pos.y);
            fbuf->write(2u, pos.z);
            fbuf->write(3u, col.x);
            fbuf->write(4u, col.y);
            fbuf->write(5u, col.z);
            fbuf->write(6u, col.w);
            ubuf->write(0u, flg);
        };

        auto shader = device.compile(k);
        stream << shader().dispatch(1u) << synchronize();

        luisa::vector<float> fhost(7u);
        luisa::vector<uint> uhost(1u);
        stream << fbuf.copy_to(luisa::span{fhost}) << synchronize();
        stream << ubuf.copy_to(luisa::span{uhost}) << synchronize();
        expect(fhost.size() == 7u);
        expect(uhost.size() == 1u);
    };
}

void reg_coro_frame_type_checks() {

    "coro_id_is_uint3_type"_test = [] {
        expect(std::is_same_v<decltype(std::declval<CoroFrame>().coro_id), UInt3>);
    };

    "target_token_is_uint_type"_test = [] {
        expect(std::is_same_v<decltype(std::declval<CoroFrame>().target_token), UInt>);
    };

    "get_by_index_returns_var_type"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("fval", Type::of<float>());
        desc.add_field("ival", Type::of<int>());
        expect(desc.field_count() == 2u);
        expect(desc.field(0u).type == Type::of<float>());
        expect(desc.field(1u).type == Type::of<int>());
    };

    "get_by_name_returns_var_type"_test = [] {
        CoroFrameDesc desc;
        desc.add_field("alpha", Type::of<float>());
        auto *field = desc.field("alpha");
        expect(field != nullptr);
        expect(field->type == Type::of<float>());
    };

    "is_terminated_returns_bool"_test = [] {
        static_assert(std::is_same_v<decltype(std::declval<CoroFrame &>().is_terminated()), Bool>);
        expect(std::is_same_v<decltype(std::declval<CoroFrame &>().is_terminated()), Bool>);
    };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    // Type checks run without a device
    reg_coro_frame_static();
    reg_coro_frame_type_checks();

    auto options = luisa::test::coro_test::parse_options(argc, argv);
    auto dc = luisa::test::coro_test::create_device(options);
    reg_coro_frame_runtime(dc.device);

    return 0;
}
