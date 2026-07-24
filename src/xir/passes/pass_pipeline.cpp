#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/early_cse.h>
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
#include <luisa/xir/passes/phi_cleanup.h>
#include <luisa/xir/passes/fuse_consecutive_buffer_reads.h>
#include <luisa/xir/passes/if_conversion.h>
#include <luisa/xir/passes/licm.h>
#include <luisa/xir/passes/indvar_simplify.h>
#include <luisa/xir/passes/loop_fusion.h>
#include <luisa/xir/passes/loop_vectorization.h>
#include <luisa/xir/passes/slp_vectorization.h>

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

void PassPipeline::_merge_record(Stats::Record &record,
                                 const Stats::Record &other) noexcept {
    LUISA_ASSERT(record.name == other.name &&
                     record.children.size() == other.children.size(),
                 "Pass pipeline group shape changed while running.");
    record.invocations += other.invocations;
    record.elapsed_ms += other.elapsed_ms;
    record.changed |= other.changed;
    record.report.merge_sum(other.report);
    for (size_t i = 0u; i < other.children.size(); ++i) {
        _merge_record(record.children[i], other.children[i]);
    }
}

PassPipeline::Stats::Record PassPipeline::_run_entry(const Entry &entry,
                                                      Module *module) noexcept {
    if (!entry.is_group) {
        luisa::Clock clock;
        PassReport report;
        auto changed = entry.run(module, report);
        return Stats::Record{
            .name = entry.name,
            .invocations = 1u,
            .elapsed_ms = clock.toc(),
            .changed = changed,
            .report = std::move(report),
            .children = {},
        };
    }
    Stats::Record record{
        .name = entry.name,
        .invocations = 0u,
        .elapsed_ms = 0.0,
        .changed = false,
        .report = {},
        .children = {},
    };
    record.children.reserve(entry.children.size());
    luisa::Clock clock;
    for (uint32_t iteration = 0u; iteration < entry.max_iterations; ++iteration) {
        auto any_changed = false;
        for (size_t i = 0u; i < entry.children.size(); ++i) {
            auto child = _run_entry(entry.children[i], module);
            any_changed |= child.changed;
            if (iteration == 0u) {
                record.children.emplace_back(std::move(child));
            } else {
                _merge_record(record.children[i], child);
            }
        }
        record.invocations++;
        if (!any_changed) { break; }
        record.changed = true;
    }
    record.elapsed_ms = clock.toc();
    return record;
}

void PassPipeline::_run_entries(luisa::span<const Entry> entries,
                                Module *module, Stats &stats) noexcept {
    for (auto &entry : entries) {
        stats.records.emplace_back(_run_entry(entry, module));
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
    // Structured loop transforms remain available as explicit passes, but are
    // excluded from default pipelines until they preserve structured ownership,
    // loop-carried PHIs, and break/continue semantics.
    p.add("licm", [](Module *m, PassReport &r) {
        auto i = licm_pass_run_on_module(m, &r);
        return i.hoisted_count > 0u;
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
    // Inlining exposes new adjacent-store chains; vectorize them before the
    // scalarizer (running later) decomposes vector ops again.
    p.add("slp-vectorization", [](Module *m, PassReport &r) {
        auto i = slp_vectorization_pass_run_on_module(m, &r);
        return i.vectorized_tree_count > 0u;
    });
    p.add("fuse-consecutive-buffer-reads", [](Module *m, PassReport &r) {
        auto i = fuse_consecutive_buffer_reads_pass_run_on_module(m, &r);
        return i.fused_group_count > 0u;
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
    // See create_basic_optimization_pipeline: unsafe structured loop transforms
    // are intentionally opt-in for now.
    p.add("licm", [](Module *m, PassReport &r) {
        auto i = licm_pass_run_on_module(m, &r);
        return i.hoisted_count > 0u;
    });
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("sccp", [](Module *m, PassReport &r) {
        auto i = sccp_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u || i.removed_branch_count > 0u;
    });
    p.add("slp-vectorization", [](Module *m, PassReport &r) {
        auto i = slp_vectorization_pass_run_on_module(m, &r);
        return i.vectorized_tree_count > 0u;
    });
    p.add("gvn", [](Module *m, PassReport &r) {
        auto i = gvn_pass_run_on_module(m, &r);
        return i.replaced_inst_count > 0u || i.removed_inst_count > 0u;
    });
    p.add("if-conversion", [](Module *m, PassReport &r) {
        auto i = if_conversion_pass_run_on_module(m, &r);
        return i.converted_diamond_count > 0u;
    });
    p.add("phi-cleanup", [](Module *m, PassReport &r) {
        auto i = phi_cleanup_pass_run_on_module(m, &r);
        return i.removed_phi_count > 0u;
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
    p.add("const-fold", [](Module *m, PassReport &r) {
        auto i = const_fold_pass_run_on_module(m, &r);
        return i.folded_inst_count > 0u;
    });
    p.add("algebraic-simplify", [alg_opts](Module *m, PassReport &r) {
        auto i = algebraic_simplify_pass_run_on_module(m, alg_opts, &r);
        return i.simplified_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    p.add("early-cse", [](Module *m, PassReport &r) {
        auto i = early_cse_pass_run_on_module(m, &r);
        return i.eliminated_inst_count > 0u;
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
    p.add("gvn", [](Module *m, PassReport &r) {
        auto i = gvn_pass_run_on_module(m, &r);
        return i.replaced_inst_count > 0u || i.removed_inst_count > 0u;
    });
    p.add("dce", [](Module *m, PassReport &r) {
        auto i = dce_pass_run_on_module(m, &r);
        return i.removed_inst_count > 0u || i.removed_block_count > 0u;
    });
    return p;
}

}// namespace luisa::compute::xir
