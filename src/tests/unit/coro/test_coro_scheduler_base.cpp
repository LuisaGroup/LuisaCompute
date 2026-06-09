#include "ut/ut.hpp"
#include <luisa/coro/coro_scheduler.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

/// Concrete stub that records the last dispatch call.
struct TestScheduler final : CoroScheduler<int, float> {
    bool called{false};
    uint3 last_size{};
    int last_int{0};
    float last_float{0.f};

    using CoroScheduler::CoroScheduler;

    void _dispatch(Stream &, uint3 size, const int &i, const float &f) noexcept override {
        called = true;
        last_size = size;
        last_int = i;
        last_float = f;
    }
};

}// namespace

void reg_coro_scheduler_base() {

    "constructor_stores_graph_and_frame_desc"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        desc.add_field("value", Type::of<float>());

        TestScheduler sched{graph, desc};

        expect(&sched.graph() == &graph);
        expect(&sched.frame_desc() == &desc);
        expect(sched.frame_desc().field_count() == 1u);
        expect(sched.frame_desc().total_size() > 0u);
    };

    "operator_paren_returns_coro_task_submitter"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        auto submitter = sched(42, 3.14f);

        // Verify the type name contains CoroTaskSubmitter
        using SubmitterType = decltype(submitter);
        static_assert(std::is_same_v<
                          SubmitterType,
                          CoroScheduler<int, float>::CoroTaskSubmitter>);
        expect(true);// compile-time check passed
    };

    "dispatch_calls_underscore_dispatch"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        Stream dummy_stream;
        uint3 size = make_uint3(16u, 8u, 1u);

        auto dispatcher = sched(42, 3.14f).dispatch(size);
        dispatcher(dummy_stream);

        expect(sched.called);
        expect(sched.last_size.x == 16u);
        expect(sched.last_size.y == 8u);
        expect(sched.last_size.z == 1u);
        expect(sched.last_int == 42);
        expect(sched.last_float == 3.14_f);
    };

    "dispatch_1d_convenience"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        Stream dummy_stream;
        auto dispatcher = sched(7, 2.71f).dispatch(32u);
        dispatcher(dummy_stream);

        expect(sched.called);
        expect(sched.last_size.x == 32u);
        expect(sched.last_size.y == 1u);
        expect(sched.last_size.z == 1u);
        expect(sched.last_int == 7);
        expect(sched.last_float == 2.71_f);
    };

    "dispatch_2d_convenience"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        Stream dummy_stream;
        auto dispatcher = sched(-1, 0.5f).dispatch(8u, 4u);
        dispatcher(dummy_stream);

        expect(sched.called);
        expect(sched.last_size.x == 8u);
        expect(sched.last_size.y == 4u);
        expect(sched.last_size.z == 1u);
    };

    "dispatch_3d_convenience"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        Stream dummy_stream;
        auto dispatcher = sched(100, 9.99f).dispatch(2u, 3u, 4u);
        dispatcher(dummy_stream);

        expect(sched.called);
        expect(sched.last_size.x == 2u);
        expect(sched.last_size.y == 3u);
        expect(sched.last_size.z == 4u);
    };

    "dispatch_with_rvalue_args_moves_correctly"_test = [] {
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        Stream dummy_stream;
        // rvalue arguments should be perfectly forwarded
        auto dispatcher = sched(1 + 2, 3.0f + 0.14f).dispatch(1u);
        dispatcher(dummy_stream);

        expect(sched.called);
        expect(sched.last_int == 3);
        expect(sched.last_float == 3.14_f);
    };

    "pure_virtual_prevents_instantiation"_test = [] {
        // CoroScheduler<int, float> itself is abstract
        static_assert(std::is_abstract_v<CoroScheduler<int, float>>);
        // Derived concrete type is not abstract
        static_assert(!std::is_abstract_v<TestScheduler>);
        expect(true);
    };

    "stream_syntax_compiles"_test = [] {
        // Verify that stream << sched(args).dispatch(size) compiles
        // by checking the type of the dispatch callable is invocable with Stream&
        CoroGraph graph;
        CoroFrameDesc desc;
        TestScheduler sched{graph, desc};

        auto dispatched = sched(10, 1.0f).dispatch(make_uint3(1u));
        using DispatchedType = decltype(dispatched);

        static_assert(std::is_invocable_v<DispatchedType, Stream &>);
        expect(true);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_scheduler_base();
    return 0;
}
