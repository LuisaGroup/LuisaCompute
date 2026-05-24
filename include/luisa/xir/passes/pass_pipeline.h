#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/functional.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class LUISA_XIR_API PassPipeline {

public:
    struct Stats {
        struct Record {
            luisa::string_view name;
            uint32_t invocations{0u};
            double elapsed_ms{0.0};
            bool changed{false};
            luisa::vector<Record> children;
        };
        luisa::vector<Record> records;
        double total_ms{0.0};

        void log(luisa::string_view pipeline_name = {}) const noexcept;
    };

private:
    struct Entry {
        luisa::string name;
        luisa::move_only_function<bool(Module *)> run;
        uint32_t max_iterations{1u};
        bool is_group{false};
        luisa::vector<Entry> children;
    };

    luisa::vector<Entry> _entries;

    static void _run_entries(luisa::span<const Entry> entries,
                             Module *module, Stats &stats) noexcept;

public:
    PassPipeline() noexcept = default;
    ~PassPipeline() noexcept = default;
    PassPipeline(PassPipeline &&) noexcept = default;
    PassPipeline &operator=(PassPipeline &&) noexcept = default;

    PassPipeline &add(luisa::string name,
                      luisa::move_only_function<bool(Module *)> pass) noexcept;

    PassPipeline &add_fixed_point(luisa::string name,
                                  PassPipeline sub,
                                  uint32_t max_iterations = 64u) noexcept;

    [[nodiscard]] Stats run(Module *module) const noexcept;
    [[nodiscard]] bool empty() const noexcept { return _entries.empty(); }
    [[nodiscard]] size_t size() const noexcept { return _entries.size(); }
};

struct OptimizationPipelineOptions {
    bool enable_fast_math{false};
};

// Phase A: basic opts on structured-CFG alloca-form (ast2xir output).
// dce, store-forward, load-elim, dce, algebraic, const-fold, dce,
// promote-ref-arg, sroa, dse, dce.
[[nodiscard]] LUISA_XIR_API PassPipeline
create_basic_optimization_pipeline(OptimizationPipelineOptions options = {}) noexcept;

// Post-inline cleanup: dce, store-forward, load-elim, dce,
// algebraic, const-fold, dce, sroa, dse, dce.
[[nodiscard]] LUISA_XIR_API PassPipeline
create_post_inline_cleanup_pipeline(OptimizationPipelineOptions options = {}) noexcept;

// SSA optimization on unstructured CFG (after destructure_cfg + mem2reg):
// algebraic, const-fold, dce, store-forward, load-elim, dse, dce.
// TODO: add sccp and cse when those passes are fully implemented.
[[nodiscard]] LUISA_XIR_API PassPipeline
create_ssa_optimization_pipeline(OptimizationPipelineOptions options = {}) noexcept;

// Post-restructure cleanup:
// dce, store-forward, load-elim, dse, algebraic, const-fold, dce.
[[nodiscard]] LUISA_XIR_API PassPipeline
create_post_restructure_cleanup_pipeline(OptimizationPipelineOptions options = {}) noexcept;

}// namespace luisa::compute::xir
