#include "ut/ut.hpp"
#include <luisa/coro/coro_scheduler.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TestScheduler final : CoroScheduler<int, float> {
    bool called{false};
    uint3 last_size{};
    int last_int{0};
    float last_float{0.f};

    void _dispatch(
        Stream &, uint3 size,
        const int &i,
        const float &f) noexcept override {
        called = true;
        last_size = size;
        last_int = i;
        last_float = f;
    }
};

Stream dummy_stream;

}// namespace

void reg_coro_scheduler_base() {

    "scheduler_stage_option_preserves_compilation_semantics"_test = [] {
        ShaderOption base{
            .enable_cache = false,
            .enable_fast_math = false,
            .enable_debug_info = true,
            .max_registers = 73u,
            .enable_scalarizer = true,
            .enable_driver_optimization = false,
            .name = "path"};
        auto staged = coro::detail::coro_scheduler_shader_option(
            base, "resume_2");
        expect(!staged.enable_cache);
        expect(!staged.enable_fast_math);
        expect(staged.enable_debug_info);
        expect(staged.max_registers == 73u);
        expect(staged.enable_scalarizer);
        expect(!staged.enable_driver_optimization);
        expect(staged.name == "path_resume_2");

        base.name.clear();
        staged = coro::detail::coro_scheduler_shader_option(
            base, "resume_2");
        expect(staged.name.empty())
            << "unnamed shaders must retain hash-based caching";
    };

    "pure_virtual_prevents_instantiation"_test = [] {
        static_assert(std::is_abstract_v<CoroScheduler<int, float>>);
        static_assert(!std::is_abstract_v<TestScheduler>);
        expect(std::is_abstract_v<CoroScheduler<int, float>>);
        expect(!std::is_abstract_v<TestScheduler>);
    };

    "operator_paren_returns_coro_scheduler_invoke"_test = [] {
        TestScheduler sched;
        auto invoke = sched(42, 3.14f);
        using InvokeType = decltype(invoke);
        static_assert(std::is_same_v<InvokeType,
                                      coro::detail::CoroSchedulerInvoke<int, float>>);
        expect(std::is_same_v<InvokeType,
                              coro::detail::CoroSchedulerInvoke<int, float>>);
    };

    "dispatch_is_lazy_called_when_invoked_with_stream"_test = [] {
        TestScheduler sched;
        uint3 size = make_uint3(16u, 8u, 1u);
        auto dispatched = sched(42, 3.14f).dispatch(size);

        // _dispatch not called until invoked with a stream
        expect(!sched.called);

        dispatched(dummy_stream);

        expect(sched.called);
        expect(sched.last_size.x == 16u);
        expect(sched.last_size.y == 8u);
        expect(sched.last_size.z == 1u);
        expect(sched.last_int == 42);
        expect(sched.last_float == 3.14_f);
    };

    "dispatch_1d_convenience"_test = [] {
        TestScheduler sched;
        sched(7, 2.71f).dispatch(32u)(dummy_stream);
        expect(sched.called);
        expect(sched.last_size.x == 32u);
        expect(sched.last_size.y == 1u);
        expect(sched.last_size.z == 1u);
        expect(sched.last_int == 7);
        expect(sched.last_float == 2.71_f);
    };

    "dispatch_2d_convenience"_test = [] {
        TestScheduler sched;
        sched(-1, 0.5f).dispatch(8u, 4u)(dummy_stream);
        expect(sched.called);
        expect(sched.last_size.x == 8u);
        expect(sched.last_size.y == 4u);
        expect(sched.last_size.z == 1u);
    };

    "dispatch_3d_convenience"_test = [] {
        TestScheduler sched;
        sched(100, 9.99f).dispatch(2u, 3u, 4u)(dummy_stream);
        expect(sched.called);
        expect(sched.last_size.x == 2u);
        expect(sched.last_size.y == 3u);
        expect(sched.last_size.z == 4u);
    };

    "dispatch_with_rvalue_args"_test = [] {
        TestScheduler sched;
        sched(1 + 2, 3.0f + 0.14f).dispatch(1u)(dummy_stream);
        expect(sched.called);
        expect(sched.last_int == 3);
        expect(sched.last_float == 3.14_f);
    };

    "stream_syntax_compiles"_test = [] {
        TestScheduler sched;
        auto dispatched = sched(10, 1.0f).dispatch(make_uint3(1u));
        expect(!sched.called);

        dummy_stream << dispatched;
        expect(sched.called);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_scheduler_base();
    return 0;
}
