#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include <luisa/xir/op.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class BasicBlock;
class Instruction;
class Value;

namespace detail {

class CoroSemanticGraph;

// A Boolean equality known on one executable CFG edge. `predicate` names a
// value-numbered, side-effect-free expression; `value` is the value that
// expression must have when the edge is taken.
struct CoroPredicateLiteral {
    size_t predicate;
    bool value;
};

struct CoroBlockPredicateKill {
    Instruction *instruction{nullptr};
    size_t ordinal{0u};
    luisa::span<const size_t> predicates;
};

// Non-mutating predicate value numbering for sparse conditional dataflow.
//
// Pure total arithmetic is numbered structurally, with an exact descriptor
// comparison after hash lookup. Memory reads, Phis, and other dynamically
// re-executed values are leaves. Re-executing such a leaf invalidates every
// predicate depending on its previous dynamic instance. This makes repeated
// tests useful within one value lifetime without confusing loop iterations.
class CoroPredicateAnalysis {

private:
    enum class TermKind : uint8_t {
        leaf,
        arithmetic
    };

    struct Term {
        TermKind kind;
        Value *leaf{nullptr};
        const luisa::compute::Type *type{nullptr};
        ArithmeticOp op{};
        luisa::vector<size_t> operands;
        luisa::vector<Instruction *> dynamic_dependencies;
    };

    struct BlockBranchPredicate {
        size_t predicate{~size_t{0u}};
        size_t true_block_id{~size_t{0u}};
        size_t false_block_id{~size_t{0u}};
        bool true_value{true};
        bool self_killed{false};
    };

    luisa::vector<Term> _terms;
    luisa::unordered_map<Value *, size_t> _leaf_terms;
    luisa::unordered_map<uint64_t, luisa::vector<size_t>> _term_buckets;
    luisa::unordered_map<Instruction *, CoroPredicateLiteral>
        _condition_literals;
    luisa::unordered_map<Instruction *, luisa::vector<size_t>>
        _predicate_kills;
    luisa::vector<luisa::vector<CoroBlockPredicateKill>> _block_kills;
    luisa::vector<BlockBranchPredicate> _block_predicates;
    luisa::unordered_set<size_t> _registered_predicates;

private:
    [[nodiscard]] luisa::optional<size_t>
    _term_for_value(Value *value) noexcept;
    [[nodiscard]] luisa::optional<CoroPredicateLiteral>
    _literal_for_condition(Value *condition) noexcept;
    void _register_predicate(size_t predicate) noexcept;

public:
    explicit CoroPredicateAnalysis(
        const CoroSemanticGraph &graph) noexcept;
    CoroPredicateAnalysis(const CoroPredicateAnalysis &) = delete;
    CoroPredicateAnalysis &operator=(const CoroPredicateAnalysis &) = delete;
    CoroPredicateAnalysis(CoroPredicateAnalysis &&) = delete;
    CoroPredicateAnalysis &operator=(CoroPredicateAnalysis &&) = delete;

    [[nodiscard]] luisa::optional<CoroPredicateLiteral>
    literal_on_edge(BasicBlock *predecessor,
                    BasicBlock *successor) const noexcept;

    [[nodiscard]] luisa::span<const size_t>
    killed_predicates(Instruction *instruction) const noexcept;

    [[nodiscard]] luisa::span<const CoroBlockPredicateKill>
    block_kills(size_t block_id) const noexcept {
        return block_id < _block_kills.size() ?
                   luisa::span<const CoroBlockPredicateKill>{
                       _block_kills[block_id]} :
                   luisa::span<const CoroBlockPredicateKill>{};
    }

    [[nodiscard]] size_t predicate_count() const noexcept {
        return _registered_predicates.size();
    }

    [[nodiscard]] size_t term_count() const noexcept {
        return _terms.size();
    }

    [[nodiscard]] size_t block_predicate(size_t block_id) const noexcept {
        return block_id < _block_predicates.size() ?
                   _block_predicates[block_id].predicate :
                   ~size_t{0u};
    }

    [[nodiscard]] bool block_predicate_self_killed(size_t block_id) const noexcept {
        return block_id < _block_predicates.size() &&
               _block_predicates[block_id].self_killed;
    }

    [[nodiscard]] luisa::optional<CoroPredicateLiteral>
    literal_on_edge_by_id(size_t pred_block_id,
                          size_t succ_block_id) const noexcept {
        if (pred_block_id >= _block_predicates.size()) {
            return luisa::nullopt;
        }
        const auto &bp = _block_predicates[pred_block_id];
        if (bp.predicate == ~size_t{0u}) {
            return luisa::nullopt;
        }
        if (succ_block_id == bp.true_block_id &&
            succ_block_id != bp.false_block_id) {
            return CoroPredicateLiteral{bp.predicate, bp.true_value};
        }
        if (succ_block_id == bp.false_block_id &&
            succ_block_id != bp.true_block_id) {
            return CoroPredicateLiteral{bp.predicate, !bp.true_value};
        }
        return luisa::nullopt;
    }
};

}// namespace detail
}// namespace luisa::compute::xir
