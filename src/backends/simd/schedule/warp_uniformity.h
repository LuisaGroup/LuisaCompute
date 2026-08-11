#pragma once

#include <unordered_map>

#include "schedule_ir.h"

namespace luisa::compute::xir {
class Function;
class Instruction;
class ThreadGroupInst;
class Value;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

// Conservative warp-relative uniformity analysis for SIMD value formation.
// Unlike XIR's SPIR-V-oriented UniformityAnalysis, this class answers whether
// one scalar value can represent all lanes in the current dynamic cohort.
class WarpUniformityAnalysis {

private:
    enum struct State {
        unknown,
        uniform,
        varying,
    };

    const xir::Function *_function{nullptr};
    std::unordered_map<const xir::Value *, State> _states;

private:
    [[nodiscard]] State _state(const xir::Value *value) const noexcept;

public:
    void clear() noexcept;
    void analyze(const xir::Function *function) noexcept;

    [[nodiscard]] ValueClass classify(
        const xir::Value *value) const noexcept;
    [[nodiscard]] bool is_uniform(
        const xir::Value *value) const noexcept {
        return classify(value) == ValueClass::uniform;
    }
};

}// namespace luisa::compute::simd::schedule
