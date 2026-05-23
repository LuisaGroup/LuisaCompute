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
        };
        luisa::vector<Record> records;
        double total_ms{0.0};
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

}// namespace luisa::compute::xir
