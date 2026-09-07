#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/passes/coro_cfg_distill.h>

#include "coro_frame_access.h"

namespace luisa::compute::xir {

class AllocaInst;
class FunctionDefinition;
class Value;

namespace detail {

class CoroReplayableValueAnalysis;

class DenseValueSet {

private:
    size_t _bit_count{0u};
    luisa::vector<uint64_t> _words;

public:
    explicit DenseValueSet(size_t bit_count = 0u) noexcept;

    [[nodiscard]] static DenseValueSet full(size_t bit_count) noexcept;
    [[nodiscard]] size_t bit_count() const noexcept { return _bit_count; }
    [[nodiscard]] size_t word_count() const noexcept { return _words.size(); }

    void set(size_t index) noexcept;
    [[nodiscard]] bool test(size_t index) const noexcept;
    void union_with(const DenseValueSet &other) noexcept;
    void intersect_with(const DenseValueSet &other) noexcept;
    void subtract(const DenseValueSet &other) noexcept;
    [[nodiscard]] bool operator==(const DenseValueSet &other) const noexcept;
    [[nodiscard]] size_t count_size() const noexcept;

    template<typename F>
    void for_each_set_bit(F &&visit) const noexcept {
        for (size_t word_index = 0u;
             word_index < _words.size(); ++word_index) {
            auto word = _words[word_index];
            while (word != 0u) {
                auto bit = static_cast<size_t>(std::countr_zero(word));
                visit(word_index * 64u + bit);
                word &= word - 1u;
            }
        }
    }
};

// One immutable global atom numbering is shared by all coroutine scopes and
// by the inter-scope liveness fixed point. Individual intra-scope solvers may
// project this domain, but their results are always embedded back into these
// coordinates before crossing a scope boundary.
class DenseValueDomain {

private:
    CoroFrameAtomDomain _atoms;

public:
    explicit DenseValueDomain(
        FunctionDefinition *definition,
        luisa::span<Value *const> designated_values = {}) noexcept;

    [[nodiscard]] size_t size() const noexcept { return _atoms.size(); }
    [[nodiscard]] luisa::optional<size_t> ssa_index(
        Value *value) const noexcept;
    [[nodiscard]] luisa::span<const CoroFrameAtomDomain::MemoryAccess>
    memory_accesses(Value *pointer) const noexcept;
    [[nodiscard]] const CoroFrameAtomDomain::Atom &atom(
        size_t index) const noexcept;
    [[nodiscard]] size_t split_alloca_count() const noexcept;
    [[nodiscard]] size_t split_atom_count() const noexcept;
    void append_indices(luisa::vector<size_t> &destination,
                        const DenseValueSet &source) const noexcept;
};

struct DenseScopeDataflowResult {
    size_t global_value_count{0u};
    luisa::vector<size_t> local_to_global;
    DenseValueSet external;
    DenseValueSet touched;
    luisa::vector<DenseValueSet> killed_at_exit;
    luisa::vector<DenseValueSet> touched_at_exit;
    size_t must_block_evaluations{0u};
    size_t may_block_evaluations{0u};

    DenseScopeDataflowResult(size_t block_count,
                             size_t global_count,
                             luisa::vector<size_t> projection) noexcept;

    [[nodiscard]] size_t local_value_count() const noexcept {
        return local_to_global.size();
    }
    [[nodiscard]] size_t fixed_point_block_evaluations() const noexcept {
        return must_block_evaluations + may_block_evaluations;
    }
    [[nodiscard]] DenseValueSet expand_to_global(
        const DenseValueSet &source) const noexcept;
};

[[nodiscard]] AllocaInst *trace_local_alloca(Value *value) noexcept;

[[nodiscard]] DenseScopeDataflowResult analyze_scope_use_def(
    const CoroCfgDistillResult::Scope &scope,
    const DenseValueDomain &value_domain,
    CoroReplayableValueAnalysis &replayable) noexcept;

void append_legacy_values(
    luisa::vector<Value *> &dst,
    const DenseValueSet &atoms,
    const DenseValueDomain &domain) noexcept;

void append_frame_value_indices(
    luisa::vector<size_t> &dst,
    const DenseValueSet &atoms,
    luisa::span<const std::pair<size_t, size_t>>
        atom_to_frame_value_range) noexcept;

}// namespace detail
}// namespace luisa::compute::xir
