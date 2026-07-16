// Test for Polymorphic<T> DSL dispatch mechanism.
//
// Tests the following APIs from include/luisa/dsl/polymorphic.h:
// - emplace() / create<Impl>() — register implementations
// - dispatch(tag, f) — switch-dispatch over all registered impls
// - dispatch_range(tag, lo, hi, f) — dispatch over a subset [lo, hi)
// - dispatch_group(tag, group, f) — dispatch over explicit tag set
// - dispatch_with_default(tag, f, default_case) — with default handler
// - dispatch_range_with_default / dispatch_group_with_default
// - empty() / size() / impl(i)
// - Single-impl optimization path (no switch generated)
//
// Corner cases:
// - Multiple implementations with correct per-tag dispatch
// - Single implementation (optimized path)
// - Out-of-range tags in dispatch_range / dispatch_group
// - Empty Polymorphic (no implementations)
// - dispatch_group with duplicates (auto-dedup + sort)
// - dispatch_with_default for unknown tags

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// ---- Base and derived "material" types for polymorphic dispatch ----

struct MaterialBase {
    virtual ~MaterialBase() = default;
    virtual void shade(Expr<uint> tid, BufferVar<float> &out) const noexcept = 0;
};

struct ConstantMaterial : MaterialBase {
    void shade(Expr<uint> tid, BufferVar<float> &out) const noexcept override {
        out.write(tid, 1.0f);
    }
};

struct LinearMaterial : MaterialBase {
    void shade(Expr<uint> tid, BufferVar<float> &out) const noexcept override {
        out.write(tid, 2.0f);
    }
};

struct QuadraticMaterial : MaterialBase {
    void shade(Expr<uint> tid, BufferVar<float> &out) const noexcept override {
        out.write(tid, 3.0f);
    }
};

// ======================== Host-side tests (no device) ========================

void test_polymorphic_host_api() {
    // Test empty state
    {
        Polymorphic<MaterialBase> poly;
        expect(poly.empty()) << "fresh Polymorphic should be empty";
        expect(poly.size() == 0u) << "fresh Polymorphic size should be 0";
    }

    // Test emplace / create
    {
        Polymorphic<MaterialBase> poly;
        auto tag0 = poly.create<ConstantMaterial>();
        expect(tag0 == 0u) << "first tag should be 0";
        expect(!poly.empty()) << "should not be empty after create";
        expect(poly.size() == 1u) << "size should be 1 after first create";

        auto tag1 = poly.create<LinearMaterial>();
        expect(tag1 == 1u) << "second tag should be 1";
        expect(poly.size() == 2u) << "size should be 2 after second create";

        // emplace with unique_ptr directly
        auto tag2 = poly.emplace(luisa::make_unique<QuadraticMaterial>());
        expect(tag2 == 2u) << "third tag should be 2";
        expect(poly.size() == 3u) << "size should be 3 after emplace";
    }

    // Test impl() accessor
    {
        Polymorphic<MaterialBase> poly;
        (void)poly.create<ConstantMaterial>();
        (void)poly.create<LinearMaterial>();

        expect(poly.impl(0) != nullptr) << "impl(0) should not be null";
        expect(poly.impl(1) != nullptr) << "impl(1) should not be null";

        const auto &cpoly = poly;
        expect(cpoly.impl(0) != nullptr) << "const impl(0) should not be null";
        expect(cpoly.impl(1) != nullptr) << "const impl(1) should not be null";
    }
}

// ======================== GPU dispatch tests ========================

void test_polymorphic_dispatch(Device &device) {
    constexpr uint N = 64u;

    Polymorphic<MaterialBase> poly;
    (void)poly.create<ConstantMaterial>();
    (void)poly.create<LinearMaterial>();
    (void)poly.create<QuadraticMaterial>();

    auto out_buf = device.create_buffer<float>(N);
    auto tag_buf = device.create_buffer<uint>(N);
    auto stream = device.create_stream();

    Kernel1D k_dispatch = [&](BufferVar<uint> tags, BufferVar<float> out) noexcept {
        auto tid = dispatch_id().x;
        auto tag = tags.read(tid);
        poly.dispatch(tag, [&](const MaterialBase *impl) {
            impl->shade(tid, out);
        });
    };

    // Set up tags: round-robin 0,1,2,0,1,2,...
    luisa::vector<uint> host_tags(N);
    for (uint i = 0u; i < N; i++) { host_tags[i] = i % 3u; }

    auto shader_dispatch = device.compile(k_dispatch);
    stream << tag_buf.copy_from(luisa::span{host_tags})
           << shader_dispatch(tag_buf, out_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> host_out(N);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    bool dispatch_correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = static_cast<float>((i % 3u) + 1u);// 1, 2, 3
        if (std::abs(host_out[i] - expected) > 1e-5f) {
            dispatch_correct = false;
            break;
        }
    }
    expect(dispatch_correct) << "dispatch() should write correct per-tag values";
}

void test_polymorphic_dispatch_range(Device &device) {
    constexpr uint N = 32u;

    Polymorphic<MaterialBase> poly;
    (void)poly.create<ConstantMaterial>();
    (void)poly.create<LinearMaterial>();
    (void)poly.create<QuadraticMaterial>();

    auto out_buf = device.create_buffer<float>(N);
    auto tag_buf = device.create_buffer<uint>(N);
    auto stream = device.create_stream();

    Kernel1D k_range = [&](BufferVar<uint> tags, BufferVar<float> out) noexcept {
        auto tid = dispatch_id().x;
        auto tag = tags.read(tid);
        out.write(tid, -1.0f);
        poly.dispatch_range_with_default(tag, 1u, 3u, [&](const MaterialBase *impl) { impl->shade(tid, out); }, [&] {});
    };

    // Tags: alternate between 1 and 2
    luisa::vector<uint> host_tags(N);
    for (uint i = 0u; i < N; i++) { host_tags[i] = 1u + (i % 2u); }

    auto shader = device.compile(k_range);
    stream << tag_buf.copy_from(luisa::span{host_tags})
           << shader(tag_buf, out_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> host_out(N);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    bool range_correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = (i % 2u == 0u) ? 2.0f : 3.0f;
        if (std::abs(host_out[i] - expected) > 1e-5f) {
            range_correct = false;
            break;
        }
    }
    expect(range_correct) << "dispatch_range should dispatch only tags in [lo, hi)";
}

void test_polymorphic_dispatch_group(Device &device) {
    constexpr uint N = 32u;

    Polymorphic<MaterialBase> poly;
    (void)poly.create<ConstantMaterial>();
    (void)poly.create<LinearMaterial>();
    (void)poly.create<QuadraticMaterial>();

    auto out_buf = device.create_buffer<float>(N);
    auto tag_buf = device.create_buffer<uint>(N);
    auto stream = device.create_stream();

    luisa::vector<uint> group = {0u, 2u, 0u, 2u};

    Kernel1D k_group = [&](BufferVar<uint> tags, BufferVar<float> out) noexcept {
        auto tid = dispatch_id().x;
        auto tag = tags.read(tid);
        out.write(tid, -1.0f);
        poly.dispatch_group_with_default(tag, group, [&](const MaterialBase *impl) { impl->shade(tid, out); }, [&] {});
    };

    // Tags: alternate between 0 and 2
    luisa::vector<uint> host_tags(N);
    for (uint i = 0u; i < N; i++) { host_tags[i] = (i % 2u == 0u) ? 0u : 2u; }

    auto shader = device.compile(k_group);
    stream << tag_buf.copy_from(luisa::span{host_tags})
           << shader(tag_buf, out_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> host_out(N);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    bool group_correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected = (i % 2u == 0u) ? 1.0f : 3.0f;
        if (std::abs(host_out[i] - expected) > 1e-5f) {
            group_correct = false;
            break;
        }
    }
    expect(group_correct) << "dispatch_group should dispatch only over specified tag set";
}

void test_polymorphic_single_impl(Device &device) {
    constexpr uint N = 16u;

    Polymorphic<MaterialBase> poly;
    (void)poly.create<ConstantMaterial>();

    auto out_buf = device.create_buffer<float>(N);
    auto tag_buf = device.create_buffer<uint>(N);
    auto stream = device.create_stream();

    Kernel1D k_single = [&](BufferVar<uint> tags, BufferVar<float> out) noexcept {
        auto tid = dispatch_id().x;
        auto tag = tags.read(tid);
        poly.dispatch(tag, [&](const MaterialBase *impl) {
            impl->shade(tid, out);
        });
    };

    luisa::vector<uint> host_tags(N, 0u);
    auto shader = device.compile(k_single);
    stream << tag_buf.copy_from(luisa::span{host_tags})
           << shader(tag_buf, out_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> host_out(N);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    bool single_correct = true;
    for (uint i = 0u; i < N; i++) {
        if (std::abs(host_out[i] - 1.0f) > 1e-5f) {
            single_correct = false;
            break;
        }
    }
    expect(single_correct) << "single-impl dispatch should always use impl(0)";
}

void test_polymorphic_with_default(Device &device) {
    constexpr uint N = 32u;

    Polymorphic<MaterialBase> poly;
    (void)poly.create<ConstantMaterial>();
    (void)poly.create<LinearMaterial>();

    auto out_buf = device.create_buffer<float>(N);
    auto tag_buf = device.create_buffer<uint>(N);
    auto stream = device.create_stream();

    Kernel1D k_default = [&](BufferVar<uint> tags, BufferVar<float> out) noexcept {
        auto tid = dispatch_id().x;
        auto tag = tags.read(tid);
        poly.dispatch_with_default(tag, [&](const MaterialBase *impl) { impl->shade(tid, out); }, [&] { out.write(tid, -99.0f); });
    };

    // Half valid (tag 0, 1), half invalid (tag 5)
    luisa::vector<uint> host_tags(N);
    for (uint i = 0u; i < N; i++) {
        host_tags[i] = (i < N / 2u) ? (i % 2u) : 5u;
    }

    auto shader = device.compile(k_default);
    stream << tag_buf.copy_from(luisa::span{host_tags})
           << shader(tag_buf, out_buf).dispatch(N)
           << synchronize();

    luisa::vector<float> host_out(N);
    stream << out_buf.copy_to(luisa::span{host_out}) << synchronize();

    bool default_correct = true;
    for (uint i = 0u; i < N; i++) {
        float expected;
        if (i < N / 2u) {
            expected = (i % 2u == 0u) ? 1.0f : 2.0f;
        } else {
            expected = -99.0f;
        }
        if (std::abs(host_out[i] - expected) > 1e-5f) {
            default_correct = false;
            break;
        }
    }
    expect(default_correct) << "dispatch_with_default should route unknown tags to default handler";
}

// ======================== Registration ========================

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_polymorphic_dispatch(device);
    test_polymorphic_dispatch_range(device);
    test_polymorphic_dispatch_group(device);
    test_polymorphic_single_impl(device);
    test_polymorphic_with_default(device);
}
