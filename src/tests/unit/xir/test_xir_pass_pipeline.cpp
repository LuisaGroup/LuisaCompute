// Test for XIR pass-pipeline ordering, nesting, fixed points, and failure propagation.

#include "ut/ut.hpp"
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>

using luisa::compute::Type;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {

    "empty_pipeline"_test = [] {
        PassPipeline p;
        expect(p.empty());
        expect(p.size() == 0u);
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.empty());
    };

    "single_pass"_test = [] {
        PassPipeline p;
        p.add("noop", [](Module *, PassReport &) { return false; });
        expect(!p.empty());
        expect(p.size() == 1u);
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].name == "noop");
        expect(stats.records[0].invocations == 1u);
        expect(!stats.records[0].changed);
        expect(stats.records[0].children.empty());
    };

    "pass_reports_changed"_test = [] {
        PassPipeline p;
        p.add("always_changes", [](Module *, PassReport &) { return true; });
        Module m;
        auto stats = p.run(&m);
        expect(stats.records[0].changed);
    };

    "multiple_passes_ordered"_test = [] {
        luisa::vector<int> order;
        PassPipeline p;
        p.add("first", [&](Module *, PassReport &) { order.push_back(1); return false; });
        p.add("second", [&](Module *, PassReport &) { order.push_back(2); return false; });
        p.add("third", [&](Module *, PassReport &) { order.push_back(3); return false; });
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.size() == 3u);
        expect(order.size() == 3u);
        expect(order[0] == 1);
        expect(order[1] == 2);
        expect(order[2] == 3);
    };

    "fixed_point_converges"_test = [] {
        int counter = 0;
        PassPipeline sub;
        sub.add("countdown", [&](Module *, PassReport &) {
            counter++;
            return counter < 3;
        });
        PassPipeline p;
        p.add_fixed_point("converge", std::move(sub), 64u);
        Module m;
        auto stats = p.run(&m);
        expect(counter == 3);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].name == "converge");
        expect(stats.records[0].invocations == 3u);
        expect(stats.records[0].changed);
        expect(stats.records[0].converged);
        expect(!stats.records[0].iteration_limit_reached);
        expect(stats.succeeded());
        expect(stats.records[0].children.size() == 1u);
        expect(stats.records[0].children[0].name == "countdown");
        expect(stats.records[0].children[0].invocations == 3u);
    };

    "fixed_point_respects_max_iterations"_test = [] {
        int counter = 0;
        PassPipeline sub;
        sub.add("infinite", [&](Module *, PassReport &) {
            counter++;
            return true;
        });
        PassPipeline p;
        p.add_fixed_point("bounded", std::move(sub), 5u);
        Module m;
        auto stats = p.run(&m);
        expect(counter == 5);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].invocations == 5u);
        expect(!stats.records[0].converged);
        expect(stats.records[0].iteration_limit_reached);
        expect(!stats.records[0].succeeded());
        expect(!stats.succeeded());
    };

    "fixed_point_zero_budget_is_reported_not_silently_converged"_test = [] {
        auto invoked = false;
        PassPipeline sub;
        sub.add("unreachable", [&](Module *, PassReport &) {
            invoked = true;
            return false;
        });
        PassPipeline p;
        p.add_fixed_point("zero-budget", std::move(sub), 0u);
        Module m;
        auto stats = p.run(&m);
        expect(!invoked);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].invocations == 0u);
        expect(!stats.records[0].converged);
        expect(stats.records[0].iteration_limit_reached);
        expect(!stats.succeeded());
    };

    "one_shot_sequence_can_change_without_false_nonconvergence"_test = [] {
        auto invocations = 0u;
        PassPipeline sub;
        sub.add("change-once", [&](Module *, PassReport &) {
            ++invocations;
            return true;
        });
        PassPipeline p;
        p.add_sequence("one-shot", std::move(sub));
        Module m;
        auto stats = p.run(&m);
        expect(invocations == 1u);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].changed);
        expect(stats.records[0].converged);
        expect(!stats.records[0].iteration_limit_reached);
        expect(stats.succeeded());
    };

    "real_dce_pass_wrapper"_test = [] {
        PassPipeline p;
        p.add("dce", [](Module *m, PassReport &r) {
            auto info = dce_pass_run_on_module(m);
            r.set("removed_inst", info.removed_inst_count);
            r.set("removed_block", info.removed_block_count);
            return info.removed_inst_count > 0 || info.removed_block_count > 0;
        });
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].name == "dce");
        expect(!stats.records[0].changed);
    };

    "stats_timing"_test = [] {
        PassPipeline p;
        p.add("sleep_pass", [](Module *, PassReport &) {
            volatile int x = 0;
            for (int i = 0; i < 100000; ++i) x += i;
            (void)x;
            return false;
        });
        Module m;
        auto stats = p.run(&m);
        expect(stats.total_ms > 0.0);
        expect(stats.records[0].elapsed_ms > 0.0);
    };

    "stats_log"_test = [] {
        int counter = 0;
        PassPipeline sub;
        sub.add("inner_a", [&](Module *, PassReport &) { counter++; return counter < 2; });
        sub.add("inner_b", [&](Module *, PassReport &) { return false; });
        PassPipeline p;
        p.add("outer_pass", [](Module *, PassReport &r) {
            r.set("foo_count", 42u);
            r.set("bar_count", 7u);
            return true;
        });
        p.add_fixed_point("group", std::move(sub), 10u);
        Module m;
        auto stats = p.run(&m);
        stats.log("test_pipeline");
        expect(stats.records.size() == 2u);
        expect(stats.records[1].children.size() == 2u);
        expect(stats.records[0].report.entries().size() == 2u);
    };

    "stats_own_pass_names"_test = [] {
        Module m;
        auto stats = [&] {
            PassPipeline pipeline;
            pipeline.add("owned-name", [](Module *, PassReport &) { return false; });
            return pipeline.run(&m);
        }();
        expect(stats.records.size() == 1u);
        expect(stats.records[0].name == "owned-name");
    };

    "nested_fixed_point_groups"_test = [] {
        Module m;
        uint32_t invocations = 0u;
        PassPipeline leaf;
        leaf.add("leaf", [&](Module *, PassReport &) {
            invocations++;
            return invocations < 2u;
        });
        PassPipeline middle;
        middle.add_fixed_point("inner", std::move(leaf), 4u);
        PassPipeline pipeline;
        pipeline.add_fixed_point("outer", std::move(middle), 4u);
        auto stats = pipeline.run(&m);
        expect(invocations == 3u);
        expect(stats.records.size() == 1u);
        expect(stats.records[0].children.size() == 1u);
        expect(stats.records[0].children[0].children.size() == 1u);
        expect(stats.records[0].children[0].children[0].name == "leaf");
        expect(stats.records[0].children[0].children[0].invocations == 3u);
    };

    "factory_basic_optimization"_test = [] {
        auto p = create_basic_optimization_pipeline({.enable_fast_math = false});
        expect(!p.empty());
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.size() == p.size());
        for (auto &&record : stats.records) {
            expect(record.name != "loop-fusion");
        }
    };

    "factory_ssa_optimization_excludes_unsafe_loop_transforms"_test = [] {
        auto p = create_ssa_optimization_pipeline({.enable_fast_math = false});
        expect(!p.empty());
        Module m;
        auto stats = p.run(&m);
        expect(stats.records.size() == p.size());
        // Structured-CFG-unsafe loop transforms stay opt-in, but
        // slp-vectorization is block-local and intentionally wired in.
        auto saw_slp = false;
        for (auto &&record : stats.records) {
            expect(record.name != "loop-fusion");
            expect(record.name != "indvar-simplify");
            expect(record.name != "loop-vectorization");
            saw_slp = saw_slp || record.name == "slp-vectorization";
        }
        expect(saw_slp);
    };

    "factory_fast_math_option_controls_radix_pow_canonicalization"_test = [] {
        auto run = [](bool enable_fast_math) noexcept {
            Module m;
            auto *function = m.create_callable(Type::of<float>());
            auto *exponent =
                function->create_value_argument(Type::of<float>());
            auto *body = function->create_body_block();
            auto *one = m.create_constant_one(Type::of<float>());
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *base = builder.call(
                Type::of<float>(), ArithmeticOp::BINARY_ADD,
                {one, one});
            auto *power = builder.call(
                Type::of<float>(), ArithmeticOp::POW,
                {base, exponent});
            auto *ret = builder.return_(power);
            auto pipeline = create_basic_optimization_pipeline(
                {.enable_fast_math = enable_fast_math});
            auto stats = pipeline.run(&m);
            expect(stats.succeeded());
            expect(ret->return_value()->isa<ArithmeticInst>());
            return static_cast<ArithmeticInst *>(ret->return_value())
                ->op();
        };
        expect(run(false) == ArithmeticOp::POW);
        expect(run(true) == ArithmeticOp::EXP2);
    };

    "pass_report_set_overwrites"_test = [] {
        PassReport r;
        r.set("foo", 1u);
        r.set("foo", 2u);
        expect(r.entries().size() == 1u);
        expect(r.entries()[0].value == 2u);
    };

    "pass_report_merge_sum"_test = [] {
        PassReport a;
        a.set("x", 1u);
        a.set("y", 2u);
        PassReport b;
        b.set("x", 3u);
        b.set("z", 4u);
        a.merge_sum(b);
        expect(a.entries().size() == 3u);
        for (auto &e : a.entries()) {
            if (e.key == "x") expect(e.value == 4u);
            if (e.key == "y") expect(e.value == 2u);
            if (e.key == "z") expect(e.value == 4u);
        }
    };
}
