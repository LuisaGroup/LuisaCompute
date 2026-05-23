#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

PassPipeline &PassPipeline::add(luisa::string name,
                                luisa::move_only_function<bool(Module *)> pass) noexcept {
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
            };
            luisa::Clock clock;
            for (uint32_t iter = 0u; iter < entry.max_iterations; ++iter) {
                bool any_changed = false;
                for (auto &child : entry.children) {
                    auto changed = child.run(module);
                    any_changed |= changed;
                }
                rec.invocations++;
                if (!any_changed) { break; }
                rec.changed = true;
            }
            rec.elapsed_ms = clock.toc();
            stats.records.emplace_back(rec);
        } else {
            luisa::Clock clock;
            auto changed = entry.run(module);
            auto elapsed = clock.toc();
            stats.records.emplace_back(Stats::Record{
                .name = entry.name,
                .invocations = 1u,
                .elapsed_ms = elapsed,
                .changed = changed,
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

}// namespace luisa::compute::xir
