#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class AllocaInst;
class FunctionDefinition;
class Value;

namespace detail {

// Immutable value-numbering domain used by coroutine liveness. Ordinary SSA
// values occupy one atom. Local aggregate pointers are interpreted in a
// type-shaped leaf-mask domain: each projection has a May set of leaves it can
// address and a Must set it definitely overwrites. The atoms are the maximal
// type subtrees on which every observed May/Must mask is uniform. This is the
// minimal tree-representable disjoint partition that preserves all transfer
// functions; static, overlapping, and nested dynamic projections therefore do
// not force unrelated sibling subaggregates into one whole-allocation atom.
class CoroFrameAtomDomain {
public:
    struct Atom {
        Value *root{nullptr};
        luisa::vector<uint32_t> access_chain;
        const Type *type{nullptr};
    };

    struct MemoryAccess {
        size_t atom_index{0u};
        // True iff storing through this pointer defines every byte represented
        // by the atom. A descendant store overlaps an enclosing atom but does
        // not cover it, so the untouched bytes remain live.
        bool covers_atom{false};
    };

private:
    luisa::vector<Atom> _atoms;
    luisa::unordered_map<Value *, size_t> _ssa_indices;
    luisa::unordered_map<Value *, luisa::vector<MemoryAccess>> _memory_accesses;
    size_t _split_alloca_count{0u};
    size_t _split_atom_count{0u};

public:
    explicit CoroFrameAtomDomain(FunctionDefinition *definition) noexcept;

    [[nodiscard]] size_t size() const noexcept { return _atoms.size(); }
    [[nodiscard]] const Atom &atom(size_t index) const noexcept {
        return _atoms[index];
    }
    [[nodiscard]] luisa::optional<size_t> ssa_index(
        Value *value) const noexcept;
    [[nodiscard]] luisa::span<const MemoryAccess> memory_accesses(
        Value *pointer) const noexcept;
    [[nodiscard]] size_t split_alloca_count() const noexcept {
        return _split_alloca_count;
    }
    [[nodiscard]] size_t split_atom_count() const noexcept {
        return _split_atom_count;
    }
};

}// namespace detail
}// namespace luisa::compute::xir
