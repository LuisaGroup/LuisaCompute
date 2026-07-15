#include "ut/ut.hpp"
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>

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
        expect(stats.total_ms >= 0.0);
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
        expect(stats.records[0].elapsed_ms >= 0.0);
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
        p.run(&m);
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
        p.run(&m);
        expect(counter == 5);
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

    "factory_basic_optimization"_test = [] {
        auto p = create_basic_optimization_pipeline({.enable_fast_math = false});
        expect(!p.empty());
        Module m;
        auto stats = p.run(&m);
        expect(stats.total_ms >= 0.0);
        for (auto &&record : stats.records) {
            expect(record.name != "loop-fusion");
        }
    };

    "factory_ssa_optimization_excludes_unsafe_loop_transforms"_test = [] {
        auto p = create_ssa_optimization_pipeline({.enable_fast_math = false});
        expect(!p.empty());
        Module m;
        auto stats = p.run(&m);
        for (auto &&record : stats.records) {
            expect(record.name != "loop-fusion");
            expect(record.name != "indvar-simplify");
            expect(record.name != "loop-vectorization");
            expect(record.name != "slp-vectorization");
        }
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
