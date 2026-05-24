#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/local_store_forward.h>
#include <luisa/xir/passes/local_load_elimination.h>
#include <luisa/xir/passes/algebraic_simplify.h>
#include <luisa/xir/passes/const_fold.h>
#include <luisa/xir/passes/promote_ref_arg.h>
#include <luisa/xir/passes/sroa.h>
#include <luisa/xir/passes/dead_store_elimination.h>
#include <luisa/xir/passes/sccp.h>
#include <luisa/xir/passes/gvn.h>

namespace luisa::compute::xir {

void PassReport::set(luisa::string_view key, uint64_t value) noexcept {
    for (auto &e : _entries) {
        if (e.key == key) {
            e.value = value;
            return;
        }
    }
    _entries.emplace_back(Entry{.key = luisa::string{key}, .value = value});
}

void PassReport::merge_max(const PassReport &other) noexcept {
    for (auto &o : other._entries) {
        bool found = false;
        for (auto &e : _entries) {
            if (e.key == o.key) {
                e.value = std::max(e.value, o.value);
                found = true;
                break;
            }
        }
        if (!found) { _entries.emplace_back(o); }
    }
}

void PassReport::merge_sum(const PassReport &other) noexcept {
    for (auto &o : other._entries) {
        bool found = false;
        for (auto &e : _entries) {
            if (e.key == o.key) {
                e.value += o.value;
                found = true;
                break;
            }
        }
        if (!found) { _entries.emplace_back(o); }
    }
}

PassPipeline &PassPipeline::add(luisa::string name,
                                luisa::move_only_function<bool(Module *, PassReport &)> pass) noexcept {
    _entries.emplace_back(Entry{
        .name = std::move(name),
        .run = std::move(pass),
        .max_iterations = 1u,
        .is_group = false,
        .children = {},
    });
    return *this;
}

PassPipeline &PassPipeline::add_fixed_point(luisa::string name,
                                            PassPipeline sub,
                                            uint32_t max_iterations) noexcept {
    _entries.emplace_back(Entry{
        .name = std::move(name),
        .run = {},
        .max_iterations = max_iterations,
        .is_group = true,
        .children = std::move(sub._entries),
    });
    return *this;
}

void PassPipeline::_run_entries(luisa::span<const Entry> entries,
                                Module *module, Stats &stats) noexcept {
    for (auto &entry : entries) {
        if (entry.is_group) {
            Stats::Record rec{
                .name = entry.name,
                .invocations = 0u,
                .elapsed_ms = 0.0,
                .changed = false,
                .report = {},
                .children = {},
            };
            rec.children.reserve(entry.children.size());
            luisa::Clock clock;
            for (uint32_t iter = 0u; iter < entry.max_iterations; ++iter) {
                bool any_changed = false;
                for (size_t ci = 0u; ci < entry.children.size(); ++ci) {
                    luisa::Clock child_clock;
                    PassReport child_report;
                    auto changed = entry.children[ci].run(module, child_report);
                    auto child_elapsed = child_clock.toc();
                    any_changed |= changed;
                    if (iter == 0u) {
                        rec.children.emplace_back(Stats::Record{
                            .name = entry.children[ci].name,
                            .invocations = 1u,
                            .elapsed_ms = child_elapsed,
                            .changed = changed,
                            .report = std::move(child_report),
                            .children = {},
                        });
                    } else {
                        rec.children[ci].invocations++;
                        rec.children[ci].elapsed_ms += child_elapsed;
                        rec.children[ci].changed |= changed;
                        rec.children[ci].report.merge_sum(child_report);
                    }
                }
                rec.invocations++;
                if (!any_changed) { break; }
                rec.changed = true;
            }
            rec.elapsed_ms = clock.toc();
            stats.records.emplace_back(std::move(rec));
        } else {
            luisa::Clock clock;
            PassReport report;
            auto changed = entry.run(module, report);
            auto elapsed = clock.toc();
            stats.records.emplace_back(Stats::Record{
                .name = entry.name,
                .invocations = 1u,
                .elapsed_ms = elapsed,
                .changed = changed,
                .report = std::move(report),
                .children = {},
            });
        }
    }
}

PassPipeline::Stats PassPipeline::run(Module *module) const noexcept {
    Stats stats;
    stats.records.reserve(_entries.size());
    luisa::Clock total_clock;
    _run_entries(_entries, module, stats);
    stats.total_ms = total_clock.toc();
    return stats;
}

namespace detail {

static void log_records(luisa::span<const PassPipeline::Stats::Record> records,
                        uint32_t depth) noexcept {
    for (auto &rec : records) {
        luisa::string indent(depth * 2u, ' ');
        if (rec.children.empty()) {
            LUISA_VERBOSE("{}[{:6.2f} ms] {} {}",
                          indent, rec.elapsed_ms, rec.name,
                          rec.changed ? "(changed)" : "");
        } else {
            LUISA_VERBOSE("{}[{:6.2f} ms] {} (x{}) {}",
                          indent, rec.elapsed_ms, rec.name,
                          rec.invocations,
                          rec.changed ? "(changed)" : "(converged)");
        }
        for (auto &e : rec.report.entries()) {
            if (e.value > 0u) {
                LUISA_VERBOSE("{}  {}: {}", indent, e.key, e.value);
            }
        }
        if (!rec.children.empty()) {
            log_records(rec.children, depth + 1u);
        }
    }
}

}// namespace detail

void PassPipeline::Stats::log(luisa::string_view pipeline_name) const noexcept {
    if (pipeline_name.empty()) {
        LUISA_VERBOSE("PassPipeline stats ({:.2f} ms total):", total_ms);
    } else {
        LUISA_VERBOSE("PassPipeline '{}' ({:.2f} ms total):", pipeline_name, total_ms);
    }
    detail::log_records(records, 1u);
}

PassPipeline create_basic_optimization_pipeline(OptimizationPipelineOptions options) noexcept {
    auto alg_opts = AlgebraicSimplifyOptions{.enable_fast_math = options.enable_fast_math};
    PassPipeline p;
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("local-store-forward", [](Module *m, PassReport &r) {
        auto i = local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("local-load-elimination", [](Module *m, PassReport &r) {
        auto i = local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("promote-ref-arg", [](Module *m, PassReport &r) {
        auto i = promote_ref_arg_pass_run_on_module(m, &r);
        return i.promoted_ref_arg_count > 0u;
    });
    p.add("sroa", [](Module *m, PassReport &r) {
        auto i = sroa_pass_run_on_module(m, {}, &r);
        return i.decomposed_alloca_count > 0u;
    });
    p.add("dead-store-elimination", [](Module *m, PassReport &r) {
        auto i = dead_store_elimination_pass_run_on_module(m, &r);
        return i.eliminated_store_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    return p;
}

PassPipeline create_post_inline_cleanup_pipeline(OptimizationPipelineOptions options) noexcept {
    auto alg_opts = AlgebraicSimplifyOptions{.enable_fast_math = options.enable_fast_math};
    PassPipeline p;
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("local-store-forward", [](Module *m, PassReport &r) {
        auto i = local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("local-load-elimination", [](Module *m, PassReport &r) {
        auto i = local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("sroa", [](Module *m, PassReport &r) {
        auto i = sroa_pass_run_on_module(m, {}, &r);
        return i.decomposed_alloca_count > 0u;
    });
    p.add("dead-store-elimination", [](Module *m, PassReport &r) {
        auto i = dead_store_elimination_pass_run_on_module(m, &r);
        return i.eliminated_store_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    return p;
}

PassPipeline create_ssa_optimization_pipeline(OptimizationPipelineOptions options) noexcept {
    auto alg_opts = AlgebraicSimplifyOptions{.enable_fast_math = options.enable_fast_math};
    PassPipeline p;
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("gvn", [](Module *m, PassReport &r) {
        auto i = gvn_pass_run_on_module(m, &r);
        return i.replaced_inst_count > 0u || i.removed_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("local-store-forward", [](Module *m, PassReport &r) {
        auto i = local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("local-load-elimination", [](Module *m, PassReport &r) {
        auto i = local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("dead-store-elimination", [](Module *m, PassReport &r) {
        auto i = dead_store_elimination_pass_run_on_module(m, &r);
        return i.eliminated_store_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    return p;
}

PassPipeline create_post_restructure_cleanup_pipeline(OptimizationPipelineOptions options) noexcept {
    auto alg_opts = AlgebraicSimplifyOptions{.enable_fast_math = options.enable_fast_math};
    PassPipeline p;
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("local-store-forward", [](Module *m, PassReport &r) {
        auto i = local_store_forward_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("local-load-elimination", [](Module *m, PassReport &r) {
        auto i = local_load_elimination_pass_run_on_module(m, &r);
        return i.removed_load_count > 0u;
    });
    p.add("dead-store-elimination", [](Module *m, PassReport &r) {
        auto i = dead_store_elimination_pass_run_on_module(m, &r);
        return i.eliminated_store_count > 0u;
    });
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    return p;
}

}// namespace luisa::compute::xir
